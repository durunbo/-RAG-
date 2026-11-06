import os
from docx import Document
from typing import List
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from src.models import DocumentBlock


def is_heading(style_name: str) -> bool:
    """
    判断段落样式是否为标题
    兼容 Word 中英文默认样式名：Heading 1 / 标题 1
    :param style_name: 段落样式名称
    :return: True=标题，False=正文
    """
    return style_name.startswith("Heading") or style_name.startswith("标题")


def get_heading_level(style_name: str) -> int:
    """
    从样式名中提取标题层级
    例如：Heading 2 → 2，标题 3 → 3
    :param style_name: 段落样式名称
    :return: 标题层级数字，默认返回 1
    """
    for part in style_name.split():
        if part.isdigit():
            return int(part)
    return 1

# ============== 文本正则标题识别辅助函数 ==============
def _is_text_chapter(text: str) -> bool:
    """纯文本判断：是否为章节标题，和PDF/切块策略规则统一"""
    text = text.strip()
    if len(text) > 40:
        return False
    import re
    return bool(re.match(r'^第[一二三四五六七八九十百千\d]+[章节篇部分]\s*', text))

def _is_text_article(text: str) -> bool:
    """纯文本判断：是否为条款标题，和PDF/切块策略规则统一"""
    text = text.strip()
    if len(text) > 40:
        return False
    import re
    # 排除纯数字、日期、金额误判
    if re.match(r'^\d+(\.\d+)+', text):
        return False
    return bool(re.match(r'^第[一二三四五六七八九十百千\d]+条\s*', text))

def _get_text_heading_level(text: str) -> int:
    """纯文本标题的层级：章=1，条=2"""
    if _is_text_chapter(text):
        return 1
    elif _is_text_article(text):
        return 2
    return 3
# ============================================================

def load_word(file_path: str) -> List[DocumentBlock]:
    """
    读取 Word 文档，识别标题层级，解析为统一的 DocumentBlock 列表
    只负责格式解析，不做任何切块逻辑
    :param file_path: Word 文件完整路径
    :return: 解析后的 DocumentBlock 列表
    """
    doc = Document(file_path)
    file_name = os.path.basename(file_path)  # 跨平台提取纯文件名

    blocks: List[DocumentBlock] = []  # 解析结果列表
    heading_path: List[str] = []      # 维护当前的章节路径栈
    order = 0                         # 全局顺序号，保证原文顺序

    # 按 Word 原始 XML 顺序遍历所有元素：段落、表格交替出现
    for element in doc.element.body:
        # ---------- 处理段落元素 ----------
        if isinstance(element, CT_P):
            # 匹配对应段落对象
            para = next(p for p in doc.paragraphs if p._element is element)
            text = para.text.strip()
            if not text:
                continue  # 跳过空段落

            style_name = para.style.name if para.style is not None else ""

            # 每个段落必须重置标题标记与层级，否则上一段的判定结果会泄漏到本段：
            # 首段若为标题，则全文所有段落都被当成标题；首段若为正文，则此变量未绑定直接抛异常
            is_title = False
            level = 3

            # 第一级：原生样式识别（优先级最高）
            if is_heading(style_name):
                is_title = True
                level = get_heading_level(style_name)
            # 第二级：文本结构兜底（仅样式不匹配时触发）
            elif _is_text_chapter(text) or _is_text_article(text):
                is_title = True
                level = _get_text_heading_level(text)

            order += 1
            if is_title:
                # 更新层级栈：截断到上一级，推入当前标题
                heading_path = heading_path[:level - 1]
                heading_path.append(text)

                block = DocumentBlock(
                    source_file=file_name,
                    block_type="heading",
                    text=text,
                    heading_path=heading_path.copy(),
                    page=None,  # Word 无原生页码，统一置空
                    order=order
                )
            else:
                # 普通正文继承当前完整层级路径
                block = DocumentBlock(
                    source_file=file_name,
                    block_type="paragraph",
                    text=text,
                    heading_path=heading_path.copy(),
                    page=None,
                    order=order
                )
            blocks.append(block)
            continue

        # ---------- 处理表格元素 ----------
        if isinstance(element, CT_Tbl):
            # 匹配对应表格对象
            table = next(t for t in doc.tables if t._tbl is element)
            order += 1

            # 表格文本格式完全保留原有逻辑，不改动
            table_text = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                table_text.append(row_text)
            table_str = "\n".join(table_text)

            # 表格继承当前层级路径，block_type 保持 table
            block = DocumentBlock(
                source_file=file_name,
                block_type="table",
                text=table_str,
                heading_path=heading_path.copy(),
                page=None,
                order=order
            )
            blocks.append(block)
            continue

    return blocks