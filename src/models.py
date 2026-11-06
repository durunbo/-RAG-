from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentBlock:
    """
    统一的文档解析块结构
    所有格式（Word/PDF/Excel）解析后都输出这个结构
    """
    source_file: str               # 来源文件名，如 "某某公司管理制度.docx"
    block_type: str                # 块类型：heading / paragraph / table
    text: str                      # 文本内容（表格可转成markdown或纯文本）
    heading_path: List[str] = field(default_factory=list)  # 所属章节路径
    page: Optional[int] = None     # 页码（Word不一定准确，PDF必填）
    order: int = 0                 # 在原文中的顺序号，保证排序


@dataclass
class Chunk:
    """
    最终切块结果，用于向量化入库
    """
    source_file: str
    heading_path: List[str]
    text: str
    chunk_index: int          # 在该文件中的切块序号
    token_count: int = 0      # 可选，token数，用于质量检查
