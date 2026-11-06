import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from src.chunk_strategies import EnhancedChunk
from src.embedding.embedding_model import EmbeddingConfig, EmbeddingModel
from src.vector_store.vector_store import VectorStore, VectorStoreConfig


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 现有 Chunk JSON 目录
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

# 测试样例文件名前缀
TEST_FILE_PREFIX = "测试用_"


def load_chunks(include_test_files: bool = False) -> tuple[list[EnhancedChunk], int]:
    """
    读取已有 JSON，并转换为 EnhancedChunk。

    默认跳过名称以“测试用_”开头的测试样例，
    避免将测试数据误写入知识库。
    """
    if not CHUNKS_DIR.is_dir():
        raise FileNotFoundError(
            f"Chunk 目录不存在：{CHUNKS_DIR}"
        )

    json_files = sorted(CHUNKS_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"Chunk 目录中没有 JSON 文件：{CHUNKS_DIR}"
        )

    selected_files = []
    skipped_files = []

    for file_path in json_files:
        if (
            not include_test_files
            and file_path.name.startswith(TEST_FILE_PREFIX)
        ):
            skipped_files.append(file_path.name)
        else:
            selected_files.append(file_path)

    if not selected_files:
        raise ValueError(
            "没有可入库的 JSON 文件。"
            "如需包含测试样例，请使用 --include-test-files。"
        )

    chunks: list[EnhancedChunk] = []

    for file_path in selected_files:
        try:
            records = json.loads(
                file_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON 格式错误：{file_path.name}，"
                f"行 {exc.lineno}，列 {exc.colno}"
            ) from exc

        if not isinstance(records, list):
            raise ValueError(
                f"JSON 顶层必须是列表：{file_path.name}"
            )

        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"{file_path.name} 第 {record_index} 条记录不是对象"
                )

            try:
                chunk = EnhancedChunk(**record)
            except TypeError as exc:
                raise ValueError(
                    f"无法将 {file_path.name} "
                    f"第 {record_index} 条记录转换为 EnhancedChunk：{exc}"
                ) from exc

            chunks.append(chunk)

    if not chunks:
        raise ValueError("没有读取到任何 Chunk，取消入库。")

    # 检查输入数据内部是否存在重复 ID
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"

        if chunk_id in seen_ids:
            duplicate_ids.add(chunk_id)

        seen_ids.add(chunk_id)

    if duplicate_ids:
        examples = ", ".join(sorted(duplicate_ids)[:5])
        raise ValueError(
            f"输入 JSON 存在重复 Chunk ID，取消入库：{examples}"
        )

    if skipped_files:
        print(f"跳过测试样例文件：{skipped_files}")

    return chunks, len(selected_files)


@dataclass
class IndexOutcome:
    """一次入库的实际结果。

    before/after 一并带出来，是为了让调用方能核对
    「接口报的条数」与「数据库真涨的条数」是不是同一件事。
    """

    submitted: int
    inserted: int
    before: int
    after: int


async def count_chunks_by_source_file(vector_store, source_file: str) -> int:
    """统计某个文件名在库里的块数。

    直接走 VectorStore 的 engine，而不是给 VectorStore 加方法：
    本阶段规定不动 VectorStore，而这只是一条只读查询，用不上它内部的任何逻辑。
    source_file 上有索引（idx_chunks_source_file），不会全表扫。
    """
    async with vector_store.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE source_file = :source_file"),
            {"source_file": source_file},
        )
        return int(result.scalar_one())


async def index_chunks(
    vector_store,
    chunks: list[EnhancedChunk],
    *,
    source_file: str | None = None,
) -> IndexOutcome:
    """把切块写进向量库，并给出可信的写入条数。

    为什么不直接用 add_chunks 的返回值：它算的是全库 count() 差值。
    两个入库任务并行时，A 的差值里会混进 B 的块，接口报的 inserted
    和实际写入对不上。传了 source_file 就按该文件计数，
    才满足「接口报告与数据库实际一致」这条验收。

    source_file 为空时退回全库差值，等价于原 CLI 行为（批量入库场景）。
    """
    if not chunks:
        return IndexOutcome(submitted=0, inserted=0, before=0, after=0)

    if source_file is None:
        before = await vector_store.count()
        await vector_store.add_chunks(chunks)
        after = await vector_store.count()
    else:
        before = await count_chunks_by_source_file(vector_store, source_file)
        await vector_store.add_chunks(chunks)
        after = await count_chunks_by_source_file(vector_store, source_file)

    return IndexOutcome(
        submitted=len(chunks),
        inserted=max(0, after - before),
        before=before,
        after=after,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="将现有 Chunk JSON 写入 PostgreSQL + pgvector 向量数据库"
    )
    parser.add_argument(
        "--include-test-files",
        action="store_true",
        help="同时导入名称以“测试用_”开头的测试样例",
    )
    args = parser.parse_args()

    settings = get_settings()

    # 读取、校验全部数据，先不写入数据库
    chunks, file_count = load_chunks(
        include_test_files=args.include_test_files
    )

    # 检查模型目录，避免开始入库后才发现路径错误
    if not settings.embedding_model_path.is_dir():
        raise FileNotFoundError(
            f"Embedding 模型目录不存在："
            f"{settings.embedding_model_path}\n"
            "请检查 RAG_EMBEDDING_MODEL_PATH 配置。"
        )


    embedding_model = EmbeddingModel(
        config=EmbeddingConfig(
            model_name_or_path=str(settings.embedding_model_path)
        )
    )

    vector_store = None

    try:
        vector_store = VectorStore(
            embedding_model=embedding_model,
            config=VectorStoreConfig(
                database_url=str(settings.database_url),
            ),
        )

        before_count = await vector_store.count()

        print(f"待入库 JSON 文件数：{file_count}")
        print(f"待入库 Chunk 数：{len(chunks)}")
        print(f"入库前向量总数：{before_count}")
        print("开始生成 Embedding 并写入 PostgreSQL……")

        # 复用已有实现，不修改 VectorStore 业务代码
        await vector_store.add_chunks(chunks)

        after_count = await vector_store.count()

        inserted = after_count - before_count
        skipped = len(chunks) - inserted

        if inserted == 0:
            print("[WARN] 所有 Chunk 已存在，未插入任何新数据。")
            print("       (SQL ON CONFLICT DO NOTHING 跳过了重复ID)")
        elif skipped > 0:
            print(f"[WARN] 部分 Chunk 已存在，跳过 {skipped} 条。")

        print("入库完成。")
        print(f"本次提交：{len(chunks)} 条")
        print(f"实际插入：{inserted} 条")
        if skipped > 0:
            print(f"跳过重复：{skipped} 条")
        print(f"入库后向量总数：{after_count}")
        print(f"数据库连接：{settings.database_url}")

    finally:
        if vector_store is not None:
            await vector_store.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())