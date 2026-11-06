"""文档入库任务表。

放在内存里的原因：第一阶段不加 Redis / Celery，任务生命周期只有几十秒，
持久化收益远小于引入一个中间件的成本。

由此带来两个必须写进文档的约束：
1. **只能单进程运行**。--workers 4 时上传落到进程 A、轮询落到进程 B，
   进程 B 的字典里没有这个任务，表现是「传了查不到」，本地永远复现不了。
2. **进程重启后旧 task_id 查不到**（返回 404），但已入库的数据不受影响。

线程模型：所有方法都是同步的，内部只持有一把 threading.Lock，锁内没有 await。
原因是 Stage 回调发生在 asyncio.to_thread 的工作线程里
（process_document 是同步函数），那里不能 await。
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class IngestStatus(str, Enum):
    """任务状态。与 HTTP 状态码无关：上传成功一律 202，进度靠轮询这个字段。"""

    PROCESSING = "processing"      # 排队或处理中，数据尚未入库
    SUCCEEDED = "succeeded"        # 全部成功，数据已入库
    NEEDS_REVIEW = "needs_review"  # 已入库，但分类置信度低，建议人工复核
    SKIPPED = "skipped"            # 无效文档 / 无可用内容，未入库
    FAILED = "failed"              # 解析或写库失败


TERMINAL_STATUSES = frozenset(
    {
        IngestStatus.SUCCEEDED,
        IngestStatus.NEEDS_REVIEW,
        IngestStatus.SKIPPED,
        IngestStatus.FAILED,
    }
)


class IngestStage(str, Enum):
    """处理阶段。取值全部由代码真实执行路径驱动，不是按时间估算的。"""

    QUEUED = "queued"
    PARSING = "parsing"
    CLEANING = "cleaning"
    CLASSIFYING = "classifying"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    FINISHED = "finished"


@dataclass
class IngestTask:
    """单个上传任务的对外状态。

    不含临时文件路径，也不含数据库连接信息：这个对象会被序列化成 HTTP 响应。
    """

    task_id: str
    owner_user_id: str
    source_file: str
    status: IngestStatus = IngestStatus.PROCESSING
    stage: IngestStage = IngestStage.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document_type: str | None = None
    chunks_submitted: int = 0
    chunks_inserted: int = 0
    error_code: str | None = None
    message: str | None = None
    failed_stage: IngestStage | None = None
    # 文件名占用位的令牌，只在本进程内使用，不出接口
    claim_token: str | None = None


_UPDATABLE_FIELDS = frozenset(IngestTask.__dataclass_fields__) - {"task_id"}


class IngestTaskStore:
    """进程内任务表 + 文件名占用位。

    两件事合在一个类里，是因为它们共享同一把锁：
    占用位的生命周期（claim → 建任务 → 终态释放）与任务状态更新在并发上传时
    互相影响，拆成两个对象反而要加两把锁、处理死锁。
    """

    def __init__(self, *, ttl_minutes: int = 60, max_records: int = 200) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, IngestTask] = {}
        # 文件名 -> (令牌, 占用时间)
        self._claims: dict[str, tuple[str, datetime]] = {}
        self._ttl = timedelta(minutes=max(1, ttl_minutes))
        self._max_records = max(1, max_records)
        # 全库写入串行化：BGE-M3 编码是 CPU 密集，与问答共用 2 个名额，
        # 两个入库任务并行会把问答拖到卡住
        self._ingest_lock = asyncio.Lock()

    @property
    def ingest_lock(self) -> asyncio.Lock:
        """入库串行锁。只在事件循环里 await，不要在工作线程里使用。"""
        return self._ingest_lock

    # ── 文件名占用位 ──────────────────────────────────────────
    def claim(self, lock_key: str) -> str | None:
        """占用文件名。成功返回令牌，已被占用返回 None。

        为什么需要它：只查数据库挡不住并发——两个请求同时传同名文件，
        都会查到 0 条，然后都往下走，最终一个成功、一个静默跳过。
        """
        token = secrets.token_hex(8)
        with self._lock:
            if lock_key in self._claims:
                return None
            self._claims[lock_key] = (token, datetime.now(timezone.utc))
            return token

    def is_claimed(self, lock_key: str) -> bool:
        with self._lock:
            return lock_key in self._claims

    def release(self, lock_key: str, token: str | None = None) -> None:
        """释放占用位。

        token 不为空时只在令牌相符时释放：防止「早已被清理的旧任务」
        在 finally 里把别人刚拿到的新占用位删掉。
        """
        with self._lock:
            current = self._claims.get(lock_key)
            if current is None:
                return
            if token is not None and current[0] != token:
                return
            self._claims.pop(lock_key, None)

    # ── 任务 ──────────────────────────────────────────────────
    def create(
        self,
        *,
        owner_user_id: str,
        source_file: str,
        claim_token: str | None = None,
    ) -> IngestTask:
        task = IngestTask(
            task_id=f"job_{secrets.token_hex(8)}",
            owner_user_id=owner_user_id,
            source_file=source_file,
            claim_token=claim_token,
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._evict_locked()
        return task

    def get_raw(self, task_id: str) -> IngestTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get(self, task_id: str, user_id: str) -> IngestTask | None:
        """按归属取任务。不属于该用户时返回 None，由路由统一回 404。

        不区分「不存在」和「不是你的」：区分了就等于告诉对方这个 task_id 真实存在。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.owner_user_id != user_id:
                return None
            return task

    def update(self, task_id: str, **fields) -> None:
        """更新任务字段。任务不存在时静默返回。

        必须静默：清理可能在后台任务还在跑的时候删掉记录，
        此时 Stage 回调再更新会抛 KeyError，直接打断整个入库流程——
        数据已经写进去了，接口却显示失败。
        """
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            logger.warning("忽略未知任务字段：%s", sorted(unknown))
        patch = {k: v for k, v in fields.items() if k in _UPDATABLE_FIELDS}
        if not patch:
            return

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.debug(
                    "任务已过期，丢弃更新 task_id=%s fields=%s", task_id, sorted(patch)
                )
                return
            for key, value in patch.items():
                setattr(task, key, value)
            task.updated_at = datetime.now(timezone.utc)

    # ── 清理 ──────────────────────────────────────────────────
    def sweep(self, now: datetime | None = None) -> int:
        """删除过期的终态任务，并回收泄漏的占用位。返回清理条数。"""
        now = now or datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            expired = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status in TERMINAL_STATUSES and now - task.updated_at > self._ttl
            ]
            for task_id in expired:
                self._tasks.pop(task_id, None)
                removed += 1

            # 占用位自愈：进程若在 finally 之前被强杀，这个文件名会永久传不上去。
            # 只回收「没有对应存活任务、且已过期」的占用位，
            # 避免把「刚 claim 还没建任务」的窗口期误清掉。
            live_tokens = {
                task.claim_token
                for task in self._tasks.values()
                if task.status not in TERMINAL_STATUSES and task.claim_token
            }
            for lock_key, (token, claimed_at) in list(self._claims.items()):
                if token in live_tokens:
                    continue
                if now - claimed_at <= self._ttl:
                    continue
                self._claims.pop(lock_key, None)

        if removed:
            logger.info("清理过期任务记录：%s 条", removed)
        return removed

    def _evict_locked(self) -> None:
        """超出上限时淘汰最老的终态任务。调用方必须已持锁。"""
        overflow = len(self._tasks) - self._max_records
        if overflow <= 0:
            return
        finished = sorted(
            (t for t in self._tasks.values() if t.status in TERMINAL_STATUSES),
            key=lambda t: t.created_at,
        )
        for task in finished[:overflow]:
            self._tasks.pop(task.task_id, None)
