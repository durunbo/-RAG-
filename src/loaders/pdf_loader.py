import os
import re
import fitz
from typing import List, Tuple, Set
from src.models import DocumentBlock


# ==========================================================
# 一、正则规则（收紧+优化，减少误判）
# ==========================================================
# 章节级标题（仅匹配标准章节，不包含普通列表项）
_CHAPTER_PATTERNS = [
    # 第X章 / 第X篇 / 第X部分
    re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部分]\s*'),
    # 第X章 标题名称（行内只有标题）
    re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部分]\s+.{2,20}$'),
]

# 条款级标题
_ARTICLE_PATTERNS = [
    # 第X条
    re.compile(r'^第[一二三四五六七八九十百千\d]+条\s*'),
    # （X）格式条款
    re.compile(r'^（[一二三四五六七八九十\d]+）\s+'),
]

# 列表项（不作为章节/条款提升层级，仅作为普通正文）
# 一、二、三、 / 1. 2. 3.  不再识别为标题，避免误判正文列表

# 组合标题：第X章 第X条 正文（允许中间任意空格）
_COMBINED_TITLE_PATTERN = re.compile(
    r'^(第[一二三四五六七八九十百千\d]+[章节篇部分])\s*(第[一二三四五六七八九十百千\d]+条)\s*(.*)',
    re.DOTALL
)

# 页码模式：匹配各种页码格式，数字可变
_PAGE_NUMBER_PATTERNS = [
    re.compile(r'第\s*\d+\s*页\s*共\s*\d+\s*页'),
    re.compile(r'第\s*\d+\s*页\s*/\s*共\s*\d+\s*页'),
    re.compile(r'^\s*-\s*\d+\s*-\s*$'),
    re.compile(r'^\s*\d+\s*$'),
]


# ==========================================================
# 二、辅助工具函数
# ==========================================================
def _is_chapter_title(text: str) -> bool:
    """判断章节标题：收紧规则，必须是标准章节格式，避免误判列表"""
    text = text.strip()
    if len(text) > 30:  # 章节标题不会太长
        return False
    for pattern in _CHAPTER_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_article_title(text: str) -> bool:
    """判断条款标题：排除纯数字、日期、金额等误判"""
    text = text.strip()
    if len(text) > 40:
        return False
    # 排除纯数字/日期/金额：1.5万元、2025.12.15
    if re.match(r'^\d+(\.\d+)+', text):
        return False
    for pattern in _ARTICLE_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_page_number(text: str) -> bool:
    """判断是否为页码行"""
    text = text.strip()
    if len(text) > 20:
        return False
    for pattern in _PAGE_NUMBER_PATTERNS:
        if pattern.fullmatch(text):
            return True
    return False


def _normalize_page_text(text: str) -> str:
    """归一化页码文本：把数字替换成占位符，用于统计重复模式"""
    if _is_page_number(text):
        return "__PAGE_MARKER__"
    return text


def _split_combined_title(text: str) -> Tuple[str, str, str]:
    """
    拆分组合标题：章节 + 条款 + 正文
    支持中间有空格、换行；非组合标题返回空字符串占位
    """
    match = _COMBINED_TITLE_PATTERN.match(text.strip())
    if match:
        chapter = match.group(1).strip()
        article = match.group(2).strip()
        content = match.group(3).strip()
        return chapter, article, content
    return '', '', text.strip()


def _detect_header_footer(
    raw_lines: List[dict],
    page_count: int,
    threshold: float = 0.7
) -> Set[str]:
    """
    改进版页眉页脚检测
    1. 按「出现的不同页面数」统计，不是按出现次数
    2. 页码先归一化再统计
    3. 只统计顶部10%/底部10%区域的文本
    返回：判定为页眉/页脚的文本集合（归一化后）
    """
    top_page_map = {}  # {文本: {出现的页面号集合}}
    bottom_page_map = {}

    for line in raw_lines:
        text = line["text"].strip()
        if not text or len(text) > 50:
            continue

        page_num = line["page"]
        page_height = line["page_height"]
        y_ratio = line["y0"] / page_height

        # 归一化页码，避免数字不同导致无法统计
        norm_text = _normalize_page_text(text)

        # 顶部10%区域
        if y_ratio < 0.1:
            if norm_text not in top_page_map:
                top_page_map[norm_text] = set()
            top_page_map[norm_text].add(page_num)

        # 底部10%区域
        elif y_ratio > 0.9:
            if norm_text not in bottom_page_map:
                bottom_page_map[norm_text] = set()
            bottom_page_map[norm_text].add(page_num)

    # 计算出现的页面占比
    header_footer = set()
    for norm_text, pages in top_page_map.items():
        if len(pages) / page_count >= threshold:
            header_footer.add(norm_text)
    for norm_text, pages in bottom_page_map.items():
        if len(pages) / page_count >= threshold:
            header_footer.add(norm_text)

    return header_footer


# ==========================================================
# 三、主入口
# ==========================================================
def load_pdf(file_path: str) -> List[DocumentBlock]:
    """
    修正版PDF解析器
    修复点：
    1. 文本块继续拆成行，逐行识别标题
    2. 页码模式单独识别，归一化后统计页眉页脚
    3. 页眉页脚按页面集合统计，只删对应位置的块
    4. 组合标题允许空格，排版兼容性更强
    5. 每个输出块order全局唯一递增
    6. 收紧标题正则，减少列表/日期/金额误判
    7. 扫描版空文本给出明确提示
    8. with语句安全打开文件
    """
    file_name = os.path.basename(file_path)

    with fitz.open(file_path) as doc:
        page_count = len(doc)

        # ---------- 第一步：提取所有行（文本块拆成行，保留位置信息） ----------
        raw_lines = []
        global_line_order = 0

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_height = page.rect.height

            # 提取文本块，按y坐标从上到下排序
            text_blocks = page.get_text("blocks", flags=fitz.TEXTFLAGS_TEXT)
            text_blocks.sort(key=lambda b: (b[1], b[0]))

            for block in text_blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                # 【关键修复】文本块拆成多行，逐行处理
                lines = text.split('\n')
                # 每行大致y坐标偏移（按行估算，保证上下顺序）
                line_height = (y1 - y0) / max(len(lines), 1)

                for i, line_text in enumerate(lines):
                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    global_line_order += 1
                    line_y0 = y0 + i * line_height

                    raw_lines.append({
                        "text": line_text,
                        "page": page_num,
                        "y0": line_y0,
                        "page_height": page_height,
                        "raw_order": global_line_order
                    })

        # ---------- 第二步：检测并过滤页眉页脚 ----------
        if page_count >= 2:
            hf_norm_set = _detect_header_footer(raw_lines, page_count)
        else:
            hf_norm_set = set()

        # 只过滤位于顶部/底部区域、且匹配页眉页脚模式的行
        # 正文里的相同文本不会被误删
        filtered_lines = []
        for line in raw_lines:
            norm_text = _normalize_page_text(line["text"].strip())
            y_ratio = line["y0"] / line["page_height"]

            # 只有在顶部/底部区域，且匹配页眉页脚模式，才过滤
            if norm_text in hf_norm_set and (y_ratio < 0.1 or y_ratio > 0.9):
                continue
            # 单独过滤页码行（无论位置，只要是纯页码标记都过滤）
            if _is_page_number(line["text"].strip()) and len(line["text"].strip()) < 15:
                continue

            filtered_lines.append(line)

        # ---------- 第三步：空文本检测（扫描版PDF提示） ----------
        if not filtered_lines:
            raise RuntimeError(
                "PDF未提取到有效文本，可能为扫描版图片文档，建议OCR后再处理"
            )

        # ---------- 第四步：逐行识别层级，生成标准块 ----------
        result_blocks = []
        block_order = 0  # 【关键修复】每个块唯一order，全局递增
        current_chapter = ""
        current_article = ""
        heading_path = []

        for line in filtered_lines:
            text = line["text"]
            page = line["page"]

            # 情况1：组合标题（章节+条款+正文在同一行）
            chapter_part, article_part, content_part = _split_combined_title(text)
            if chapter_part and article_part:
                # 更新章节
                if chapter_part != current_chapter:
                    current_chapter = chapter_part
                    heading_path = [current_chapter]
                    block_order += 1
                    result_blocks.append(DocumentBlock(
                        source_file=file_name,
                        block_type="heading",
                        text=chapter_part,
                        heading_path=heading_path.copy(),
                        page=page,
                        order=block_order
                    ))

                # 更新条款
                current_article = article_part
                heading_path = [current_chapter, current_article]
                block_order += 1
                result_blocks.append(DocumentBlock(
                    source_file=file_name,
                    block_type="heading",
                    text=article_part,
                    heading_path=heading_path.copy(),
                    page=page,
                    order=block_order
                ))

                # 剩余正文
                if content_part:
                    block_order += 1
                    result_blocks.append(DocumentBlock(
                        source_file=file_name,
                        block_type="paragraph",
                        text=content_part,
                        heading_path=heading_path.copy(),
                        page=page,
                        order=block_order
                    ))
                continue

            # 情况2：单独章节标题
            if _is_chapter_title(text):
                current_chapter = text.strip()
                current_article = ""
                heading_path = [current_chapter]
                block_order += 1
                result_blocks.append(DocumentBlock(
                    source_file=file_name,
                    block_type="heading",
                    text=text.strip(),
                    heading_path=heading_path.copy(),
                    page=page,
                    order=block_order
                ))
                continue

            # 情况3：单独条款标题
            if _is_article_title(text):
                current_article = text.strip()
                heading_path = [current_chapter, current_article] if current_chapter else [current_article]
                block_order += 1
                result_blocks.append(DocumentBlock(
                    source_file=file_name,
                    block_type="heading",
                    text=text.strip(),
                    heading_path=heading_path.copy(),
                    page=page,
                    order=block_order
                ))
                continue

            # 情况4：普通正文（继承当前层级）
            block_order += 1
            result_blocks.append(DocumentBlock(
                source_file=file_name,
                block_type="paragraph",
                text=text.strip(),
                heading_path=heading_path.copy(),
                page=page,
                order=block_order
            ))

    return result_blocks
