# import os
# import sys
# import json
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)
#
# from src.loaders import load_document
# from src.chunker import TextChunker
#
#
# def main():
#     """
#     文档处理主流程
#     1. 扫描 data/documents 下的所有文件
#     2. 根据格式调用对应解析器，生成统一 DocumentBlock
#     3. 调用通用切块器生成 Chunk
#     4. 将结果保存为 JSON 到 data/chunks 目录
#     """
#     # 路径配置
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     doc_dir = os.path.join(base_dir, "data", "documents")    # 原始文档目录
#     chunk_dir = os.path.join(base_dir, "data", "chunks")      # 切块结果输出目录
#
#     # 自动创建输出目录，不存在则新建
#     os.makedirs(chunk_dir, exist_ok=True)
#
#     # 初始化切块器，可自定义最大字符数
#     chunker = TextChunker(max_chunk_len=500)
#
#     # 遍历原始文档目录下的所有文件
#     for filename in os.listdir(doc_dir):
#         file_path = os.path.join(doc_dir, filename)
#         # 跳过子目录，只处理文件
#         if not os.path.isfile(file_path):
#             continue
#
#         # 过滤不支持的文件格式
#         ext = os.path.splitext(filename)[1].lower()
#         if ext not in [".docx", ".doc"]:
#             print(f"跳过不支持文件：{filename}")
#             continue
#
#         print(f"正在处理：{filename}")
#         try:
#             # 第一步：解析文档为统一结构
#             blocks = load_document(file_path)
#             # 第二步：执行切块
#             chunks = chunker.chunk(blocks)
#             # 第三步：保存为 JSON 文件
#             output_path = os.path.join(chunk_dir, f"{os.path.splitext(filename)[0]}.json")
#             with open(output_path, "w", encoding="utf-8") as f:
#                 json.dump([c.__dict__ for c in chunks], f, ensure_ascii=False, indent=2)
#             print(f"处理完成，输出 {len(chunks)} 个块 → {output_path}")
#         except Exception as e:
#             # 单个文件失败不影响整体流程，打印错误信息
#             print(f"处理失败 {filename}: {str(e)}")
#
#
# if __name__ == "__main__":
#     main()
#
#

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

# 项目根目录注入，解决跨目录导入问题
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.loaders import load_document
from src.cleaner import clean_blocks, CleanConfig, print_report
from src.document_classifier import classify_document
from src.chunk_strategies import EnhancedChunk, get_chunks_by_type


@dataclass
class ProcessedDocument:
    """一份文档走完解析→清洗→分类→切块后的全部结果。

    只装数据、不碰数据库、不写 JSON：
    CLI 与 HTTP 上传共用这一条管线，两边都从这里取结果、各自决定怎么落地。
    """

    source_file: str
    document_type: str = ""
    secondary_types: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    needs_review: bool = False
    is_ignore: bool = False
    reason: str = ""                       # 跳过原因，直接进接口的 message
    chunks: list[EnhancedChunk] = field(default_factory=list)
    raw_block_count: int = 0               # 解析出来的原始块数
    block_count: int = 0                   # 清洗后有效块数
    clean_report: Optional[object] = None  # 供 CLI 打印清洗报告


def process_document(
    file_path: str,
    on_stage: Optional[Callable[[str], None]] = None,
    max_len: int = 500,
) -> ProcessedDocument:
    """解析 → 清洗 → 分类 → 切块。不写 JSON、不碰数据库。

    on_stage 会在每个阶段真正开始时被调用一次，取值 "parsing" / "cleaning" /
    "classifying" / "chunking"。它跑在调用方的线程里（HTTP 上传时是
    asyncio.to_thread 的工作线程），因此回调本身必须是同步的。

    异常不做捕获：解析失败就是失败，由调用方决定是打印还是标记任务失败。
    唯一「不算异常」的情况是分类判定为无效文档、或切块策略不支持，
    这两种写进返回值，语义是「处理完成但跳过」。
    """
    filename = os.path.basename(file_path)
    doc = ProcessedDocument(source_file=filename)

    def _notify(stage: str) -> None:
        if on_stage is not None:
            on_stage(stage)

    # 1. 格式解析
    _notify("parsing")
    blocks = load_document(file_path)
    doc.raw_block_count = len(blocks)

    # 2. 文本清洗
    _notify("cleaning")
    blocks, clean_report = clean_blocks(blocks, CleanConfig())
    doc.clean_report = clean_report
    doc.block_count = len(blocks)

    # 3. 文档类型识别
    _notify("classifying")
    classify_result = classify_document(file_path, blocks)
    doc.document_type = classify_result.primary_type
    doc.secondary_types = list(classify_result.secondary_types)
    doc.scores = dict(classify_result.scores)
    doc.needs_review = classify_result.needs_review

    if classify_result.is_ignore:
        doc.is_ignore = True
        doc.reason = (
            classify_result.matched_evidence[0]
            if classify_result.matched_evidence
            else "无效文档，自动跳过"
        )
        return doc

    # 4. 按类型切块
    _notify("chunking")
    try:
        doc.chunks = get_chunks_by_type(
            classify_result.primary_type, blocks, max_len=max_len
        )
    except ValueError as exc:
        # 未知类型不自动兜底（get_chunks_by_type 的设计），这里如实标跳过
        doc.is_ignore = True
        doc.reason = f"切块策略不支持：{exc}"

    return doc


def main():
    """
    统一文档处理入口
    流程：扫描文件 → 格式解析 → 文本清洗 → 类型识别 → 对应切块策略 → 保存结果
    兼容 Word/PDF，集成清洗、分类、分类型切块全部能力
    """
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_dir = os.path.join(base_dir, "data", "documents")
    chunk_dir = os.path.join(base_dir, "data", "chunks")

    # 自动创建输出目录
    os.makedirs(chunk_dir, exist_ok=True)

    # 遍历原始文档目录
    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # 过滤Word临时锁定文件、隐藏文件
        if filename.startswith("~$") or filename.startswith(".~") or filename.startswith("."):
            print(f"跳过临时文件：{filename}")
            continue

        ext = os.path.splitext(filename)[1].lower()
        # 仅支持 docx 和 PDF，旧版 .doc 二进制文件不支持
        if ext not in [".docx", ".pdf"]:
            print(f"跳过不支持格式：{filename}")
            continue

        print(f"\n正在处理：{filename}")
        try:
            # 解析 → 清洗 → 分类 → 切块，与 HTTP 上传共用同一条管线
            doc = process_document(file_path, max_len=500)

            print(f"  解析完成，共 {doc.raw_block_count} 个块")
            print_report(doc.clean_report, filename)
            print(f"  清洗完成，剩余 {doc.block_count} 个有效块")

            if doc.is_ignore:
                print(f"  跳过处理：{doc.reason}")
                continue

            print(f"  识别主类型：{doc.document_type}")
            print(f"  次要标签：{doc.secondary_types}")
            if doc.needs_review:
                print(f"  ⚠️ 建议人工复核，各类型得分：{doc.scores}")
            print(f"  切块完成，共 {len(doc.chunks)} 个块")

            # 保存切块结果为JSON
            output_path = os.path.join(
                chunk_dir, f"{os.path.splitext(filename)[0]}.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [c.__dict__ for c in doc.chunks],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  已保存 → {output_path}")

        except Exception as e:
            # 单个文件失败不中断整体批量处理
            print(f"  处理失败：{str(e)}")


if __name__ == "__main__":
    main()
