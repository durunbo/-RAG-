"""上传文件的落地、校验与后台入库编排。

分工：本模块负责「HTTP 世界 → 文件世界 → 入库管线」的搬运；
解析 / 清洗 / 分类 / 切块的规则全部在 src/ 里，这里一行都不改。
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import UploadFile

from app.services.document_tasks import IngestStage, IngestStatus, IngestTaskStore
from src.ingest import process_document
from src.ingest_index_chunks import index_chunks

if TYPE_CHECKING:  # 只为类型标注，避免 services 与 core 互相拖入
    from app.core.resources import AppResources

logger = logging.getLogger(__name__)

# 文件名最大长度（字符）。Windows 单段路径上限 255，留出 uuid 目录的余量
_MAX_FILENAME_CHARS = 120

# 逐块读取的大小。用固定块而不是 file.read()：后者会先把整个文件读进内存
# 再判断大小，一个 2 GB 的文件会先把进程打爆
_READ_CHUNK_BYTES = 1024 * 1024

# 校验文件头用的字节数
_HEADER_BYTES = 1024

# 路径分隔符与 Windows 保留字符
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')


class UploadRejected(Exception):
    """上传在「受理阶段」被拒绝。路由把它翻译成 4xx，不创建任务。"""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def sanitize_filename(raw: str) -> str:
    """把用户提交的文件名清洗成可安全使用的单段路径。

    三步：Unicode 归一 → 去掉目录部分 → 替换危险字符。

    归一放在最前面：macOS 上传的中文文件名是 NFD（字符拆开存），Windows 是
    NFC（合成形式），肉眼一样、字节不同。不归一的话同一份文件从两端各传一次，
    库里就会有两份，问答时命中同一段话两次，答案里出现重复引用。

    最后必须断言，而不是只做替换：只替换不断言时 ".." 这类值仍可能残留，
    临时目录 / ".." 会拼到目录外面，变成往服务器任意位置写文件。
    """
    name = unicodedata.normalize("NFC", raw or "")

    # basename 在 Windows 与 POSIX 上只认各自的分隔符，
    # 这里两种都切一遍，避免 "\\evil" 在 Linux 上被当成普通文件名
    name = name.replace("\\", "/").split("/")[-1]
    name = _UNSAFE_CHARS.sub("_", name).strip().strip(".")

    if not name or name in {".", ".."}:
        raise ValueError("文件名为空或非法")

    stem, dot, suffix = name.rpartition(".")
    if dot and stem:
        # 超长时截断主体、保留扩展名，否则截断后可能丢掉 .pdf
        keep = _MAX_FILENAME_CHARS - len(suffix) - 1
        name = f"{stem[: max(1, keep)]}.{suffix}"
    else:
        name = name[:_MAX_FILENAME_CHARS]

    # 断言：清洗结果只能是单段文件名
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("文件名非法")
    if len(name) > _MAX_FILENAME_CHARS:
        raise ValueError("文件名过长")

    return name


def validate_extension(filename: str, allowed: frozenset[str]) -> str:
    """按扩展名白名单校验，返回小写扩展名。"""
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise UploadRejected(
            415,
            "UNSUPPORTED_FILE_TYPE",
            f"只支持 {', '.join(sorted(allowed))} 格式，当前为 {ext or '无扩展名'}",
        )
    return ext


def matches_signature(header: bytes, ext: str) -> bool:
    """校验文件头魔数。

    只看扩展名挡不住「.exe 改名 .pdf」：那种文件会一路走到解析阶段，
    最后报成 PARSE_FAILED，日志里分不清是真损坏还是伪装。
    """
    if ext == ".pdf":
        return b"%PDF-" in header[:_HEADER_BYTES]
    if ext == ".docx":
        # docx 是 zip 容器。空 zip（PK\\x05\\x06）不是有效 docx，不放行
        return header.startswith(b"PK\x03\x04")
    return False


async def save_upload_to_temp(
    upload: UploadFile,
    *,
    dest_dir: Path,
    filename: str,
    max_bytes: int,
) -> tuple[Path, int, bytes]:
    """把上传内容写到受控临时文件。返回 (路径, 字节数, 前 1 KB)。

    路径完全由服务端拼：dest_dir/<uuid>/<清洗后的原文件名>。
    目录名是 uuid，用户输入逃不出这一层；文件名保留原名是因为
    classify_document 会拿 basename 当证据打分（见 src/document_classifier.py），
    改成 uuid 会让分类降级，进而走错切块策略——接口显示成功，回答质量却下降。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_dir = dest_dir / secrets.token_hex(16)
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / filename

    total = 0
    header = b""
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadRejected(
                        413,
                        "FILE_TOO_LARGE",
                        f"文件超过上限 {max_bytes // (1024 * 1024)} MB",
                    )
                if len(header) < _HEADER_BYTES:
                    header += chunk[: _HEADER_BYTES - len(header)]
                handle.write(chunk)
    except BaseException:
        # 半截文件必须删掉：留着既占磁盘，又会让后续排查误以为文件完整
        remove_temp_file(target)
        raise

    if total == 0:
        remove_temp_file(target)
        raise UploadRejected(400, "EMPTY_FILE", "上传的文件是空的")

    return target, total, header


def remove_temp_file(path: Path | None) -> None:
    """删掉上传的临时文件及其 uuid 目录。失败只记日志，不影响主流程。"""
    if path is None:
        return
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        logger.warning("临时文件清理失败：%s", path, exc_info=True)


def purge_stale_uploads(dest_dir: Path, older_than_minutes: int = 60) -> int:
    """清理残留的上传文件，返回清理项数。

    正常路径会在 finally 里删掉，但 --reload 重启、Ctrl+C 强杀、断电都不会
    执行那一步。不清理的话残留会慢慢把磁盘占满，最后所有上传都失败，
    报的错还可能是别的原因。
    """
    if not dest_dir.is_dir():
        return 0

    deadline = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
    removed = 0
    for child in dest_dir.iterdir():
        try:
            mtime = datetime.fromtimestamp(child.stat().st_mtime, timezone.utc)
            if mtime > deadline:
                continue
            if child.is_dir():
                for item in child.iterdir():
                    if item.is_file():
                        item.unlink(missing_ok=True)
                child.rmdir()
            else:
                child.unlink(missing_ok=True)
            removed += 1
        except OSError:
            logger.warning("残留上传文件清理失败：%s", child, exc_info=True)
    return removed


_STAGE_ERROR_CODES = {
    IngestStage.QUEUED: "INGEST_FAILED",
    IngestStage.PARSING: "PARSE_FAILED",
    IngestStage.CLEANING: "CLEAN_FAILED",
    IngestStage.CLASSIFYING: "CLASSIFY_FAILED",
    IngestStage.CHUNKING: "CHUNK_FAILED",
    IngestStage.INDEXING: "INDEX_FAILED",
    IngestStage.FINISHED: "INGEST_FAILED",
}

# 出现这些串说明异常文本里裹着 SQL 或驱动细节。
# 实测 VectorStoreError 会把整条 INSERT 语句拼进 message，
# 原样回给调用方等于把表结构暴露出去——只留在服务端日志里。
_SQL_LEAK_MARKERS = (
    "[SQL:",
    "asyncpg",
    "sqlalchemy",
    "INSERT INTO",
    "DELETE FROM",
    "SELECT ",
)


def _safe_message(exc: BaseException, tmp_path: Path | None) -> str:
    """构造可回给调用方的失败说明。

    只给「异常类型 + 截断后的说明」，并把临时文件绝对路径换成 <upload>。
    堆栈、SQL、连接串一律不出接口。
    """
    text = str(exc).strip()
    if any(marker in text for marker in _SQL_LEAK_MARKERS):
        return f"{type(exc).__name__}：数据库操作失败，详情见服务端日志"

    if tmp_path is not None:
        text = text.replace(str(tmp_path), "<upload>")
        text = text.replace(str(tmp_path.parent), "<upload>")
    text = text.replace("\n", " ")[:200]
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


async def run_ingest_task(
    *,
    task_id: str,
    tmp_path: Path,
    source_file: str,
    claim_key: str,
    claim_token: str | None,
    resources: "AppResources",
    store: IngestTaskStore,
) -> None:
    """后台任务：解析 → 清洗 → 分类 → 切块 → 写库。

    这是 async 函数，由 BackgroundTasks 在事件循环里调用。同步的解析与切块
    必须整段丢进 asyncio.to_thread，否则一个 50 页 PDF 会把整个服务卡住几十秒，
    所有问答一起超时。
    """
    stage = IngestStage.QUEUED
    try:
        # 排队等待：上一次入库还没跑完时这里会停住，期间接口如实返回 queued
        async with store.ingest_lock:
            stage = IngestStage.PARSING
            store.update(task_id, stage=stage)

            doc = await asyncio.to_thread(
                process_document,
                str(tmp_path),
                lambda name: store.update(task_id, stage=IngestStage(name)),
            )

            if doc.is_ignore:
                store.update(
                    task_id,
                    status=IngestStatus.SKIPPED,
                    stage=IngestStage.FINISHED,
                    document_type=doc.document_type or None,
                    message=doc.reason or "文档被判定为无效，未入库",
                )
                return

            if not doc.chunks:
                store.update(
                    task_id,
                    status=IngestStatus.SKIPPED,
                    stage=IngestStage.FINISHED,
                    document_type=doc.document_type or None,
                    message="清洗后没有可入库的内容",
                )
                return

            stage = IngestStage.INDEXING
            store.update(
                task_id,
                stage=stage,
                document_type=doc.document_type or None,
                chunks_submitted=len(doc.chunks),
            )

            outcome = await index_chunks(
                resources.vector_store, doc.chunks, source_file=source_file
            )

            if outcome.inserted != len(doc.chunks):
                # 只可能是并发写入撞了主键。如实报数，不粉饰成成功：
                # 用户看到「失败」会重传，而重传会被 409 挡住，
                # 结果既传不上去，也不知道哪些块其实已经进库了。
                store.update(
                    task_id,
                    status=IngestStatus.FAILED,
                    failed_stage=stage,
                    error_code="INDEX_INCOMPLETE",
                    chunks_inserted=outcome.inserted,
                    message=(
                        f"提交 {len(doc.chunks)} 条，实际写入 {outcome.inserted} 条。"
                        "已写入的数据不会自动清理，请勿重传，联系管理员处理"
                    ),
                )
                return

            store.update(
                task_id,
                status=(
                    IngestStatus.NEEDS_REVIEW
                    if doc.needs_review
                    else IngestStatus.SUCCEEDED
                ),
                stage=IngestStage.FINISHED,
                chunks_inserted=outcome.inserted,
                message=(
                    "已入库，但分类置信度低，建议核对文件名与内容是否一致"
                    if doc.needs_review
                    else None
                ),
            )

    except asyncio.CancelledError:
        store.update(
            task_id,
            status=IngestStatus.FAILED,
            failed_stage=stage,
            error_code="TASK_CANCELLED",
            message="服务关闭或任务被取消，数据可能未完整写入",
        )
        raise
    except Exception as exc:
        logger.exception(
            "文档入库失败 task_id=%s file=%s stage=%s",
            task_id,
            source_file,
            stage.value,
        )
        store.update(
            task_id,
            status=IngestStatus.FAILED,
            failed_stage=stage,
            error_code=_STAGE_ERROR_CODES.get(stage, "INGEST_FAILED"),
            message=_safe_message(exc, tmp_path),
        )
    finally:
        remove_temp_file(tmp_path)
        store.release(claim_key, claim_token)
