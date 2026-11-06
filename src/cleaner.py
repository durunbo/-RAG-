#通用文本清洗器
import re
import unicodedata
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from src.models import DocumentBlock
from src.text_normalize import iter_unmapped_radicals, restore_cjk_radicals


# ==========================================================
# 配置与常量区：所有规则集中管理，不写死在函数里
# ==========================================================

@dataclass
class CleanConfig:
    """清洗配置：只控制清洗规则，不涉及业务逻辑"""
    normalize_unicode: bool = True  # Unicode标准化
    repair_pdf_chars: bool = True  # PDF异体字修复
    remove_emoji: bool = True  # 删除表情符号
    remove_control_chars: bool = True  # 删除不可见控制字符
    remove_header_footer: bool = True  # 自动过滤重复页眉页脚
    compress_blank_lines: bool = True  # 压缩连续空行
    preserve_structure: bool = True  # 保留换行结构（不压成空格）
    warn_unknown_chars: bool = True  # 未知字符记录警告


@dataclass
class CleanReport:
    """清洗报告：可追溯、可统计"""
    total_blocks: int = 0
    replaced_chars: int = 0
    removed_emojis: int = 0
    removed_headers: int = 0
    removed_footers: int = 0
    removed_empty_blocks: int = 0
    warnings: List[str] = field(default_factory=list)


# 已确认的PDF异体字映射表：已统一收敛到 src/text_normalize.py
# （原表 18 条里 16 条属于康熙部首，NFKC 可直接还原；剩下 4 条已并入确认表）

# 精准Emoji正则：只删除明确的表情符号，保留业务符号
# 修复原方案：大范围删除导致 ≤ ≥ ℃ % ① 等有用符号丢失的问题
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # 主要表情符号区
    "\U0001F000-\U0001F02F"  # 麻将扑克等
    "\U0001F100-\U0001F1FF"  # 带圈字母数字（PDF 里常见的 🆔 之类图标）
    "\U0001F600-\U0001F64F"  # 面部表情
    "\U0001F680-\U0001F6FF"  # 交通与符号
    "\U0001F900-\U0001F9FF"  # 补充表情
    "\u2702-\u27B0"  # Dingbats装饰
    "]+",
    flags=re.UNICODE
)

# 半角标点/半角假名区：PDF 提取常把「、。｢｣」取成半角形（､｡｢｣）。
# 只归这一段，不动全角字母数字（ＡＢＣ、１２３）——那些在 U+FF01–FF5E，
# 是企业文档里有意义的写法，转成 ASCII 属于风格改动，不属于纠错。
_HALFWIDTH_PUNCT_PATTERN = re.compile(r'[\uff61-\uff9f]')

# 控制字符：保留换行(\n)、回车(\r)、制表符(\t)
# U+FE00–U+FE0F 是变体选择符：零宽修饰，PDF 取出的 emoji 常带一个，必须一并清掉
_CONTROL_CHARS_PATTERN = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\u200d\ufe00-\ufe0f\ufeff]'
)


# ==========================================================
# 第一层：安全字符清洗（不改变语义、不破坏结构）
# ==========================================================

def _normalize_characters(text: str, config: CleanConfig, report: CleanReport) -> str:
    """
    第一层：字符级标准化
    只做安全、无歧义的字符处理，不判断业务含义
    """
    if not text:
        return text

    # 1. Unicode标准化（兼容分解→标准等价）
    if config.normalize_unicode:
        text = unicodedata.normalize('NFC', text)

    # 2. 半角标点归一（､ → 、，｡ → 。）：PDF 提取噪音，不涉及语义
    if config.normalize_unicode:
        text = _HALFWIDTH_PUNCT_PATTERN.sub(
            lambda m: unicodedata.normalize('NFKC', m.group(0)), text
        )

    # 3. PDF 部首字符还原（康熙部首 / CJK 部首补充 → 汉字）
    #    这一步是「同一个问题一个答一个不答」的根因修复：
    #    PDF 取出的「1 个⽉内」还原成「1 个月内」后，逐字核验才能命中。
    if config.repair_pdf_chars:
        text, replaced = restore_cjk_radicals(text)
        report.replaced_chars += replaced
        if config.warn_unknown_chars:
            # 还原后仍留在部首区的字符，说明确认表还没收录，需要人工确认后补
            for char in iter_unmapped_radicals(text):
                report.warnings.append(
                    f"未知部首字符：{char} (U+{ord(char):04X})"
                )

    # 4. 精准删除Emoji，保留业务符号
    if config.remove_emoji:
        before_len = len(text)
        text = _EMOJI_PATTERN.sub('', text)
        report.removed_emojis += before_len - len(text)

    # 5. 删除不可见控制字符（保留换行、制表符）
    if config.remove_control_chars:
        text = _CONTROL_CHARS_PATTERN.sub('', text)

    return text


# ==========================================================
# 第二层：页面布局清洗（处理页眉页脚、空块、PDF断行）
# ==========================================================

def _detect_repeated_lines(blocks: List[DocumentBlock], position: str = 'top', threshold: float = 0.7) -> set:
    """
    自动检测重复出现的页眉/页脚文本
    原理：统计每页顶部/底部的文本，出现频率超过阈值则判定为页眉/页脚
    :param position: top=页眉检测，bottom=页脚检测
    """
    page_lines = {}

    # 按页分组，提取每页顶部/底部的文本
    for block in blocks:
        if block.page is None:
            continue
        if block.page not in page_lines:
            page_lines[block.page] = []
        page_lines[block.page].append(block.text.strip())

    if len(page_lines) < 2:
        return set()

    # 统计每行出现的页数占比
    line_count = {}
    total_pages = len(page_lines)

    for page, lines in page_lines.items():
        if not lines:
            continue
        # 取每页前2行（页眉）或后2行（页脚）
        target_lines = lines[:2] if position == 'top' else lines[-2:]
        for line in target_lines:
            if not line:
                continue
            line_count[line] = line_count.get(line, 0) + 1

    # 超过阈值的判定为重复页眉/页脚
    repeated = set()
    for line, count in line_count.items():
        if count / total_pages >= threshold and len(line) < 50:
            repeated.add(line)

    return repeated


def _clean_layout(blocks: List[DocumentBlock], config: CleanConfig, report: CleanReport) -> List[DocumentBlock]:
    """
    第二层：布局级清洗
    处理页眉页脚、空块、连续空行，依赖页面信息
    """
    if not blocks:
        return blocks

    result = []

    # 1. 自动检测并过滤页眉页脚
    if config.remove_header_footer:
        headers = _detect_repeated_lines(blocks, 'top')
        footers = _detect_repeated_lines(blocks, 'bottom')
    else:
        headers = set()
        footers = set()

    for block in blocks:
        text = block.text.strip()

        # 过滤页眉
        if text in headers:
            report.removed_headers += 1
            continue

        # 过滤页脚
        if text in footers:
            report.removed_footers += 1
            continue

        # 过滤纯空块
        if not text:
            report.removed_empty_blocks += 1
            continue

        # 压缩连续空行（保留单换行，不破坏结构）
        if config.compress_blank_lines:
            text = re.sub(r'\n{3,}', '\n\n', text)

        # 返回新对象，不修改原输入
        result.append(DocumentBlock(
            source_file=block.source_file,
            block_type=block.block_type,
            text=text,
            heading_path=block.heading_path.copy(),
            page=block.page,
            order=block.order
        ))

    return result


# ==========================================================
# 第三层：结构规整（保留段落、标题、问答、列表结构）
# ==========================================================

def _normalize_structure(text: str, config: CleanConfig) -> str:
    """
    第三层：结构级规整
    只做行首行尾空格清理，绝对不把换行压成空格
    完整保留：标题层级、问答结构、列表编号、表格分隔、条款边界
    """
    if not config.preserve_structure:
        return text

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 只清理每行首尾空格，保留行内空格
        cleaned = line.strip()
        cleaned_lines.append(cleaned)

    return '\n'.join(cleaned_lines)


# ==========================================================
# 统一对外接口
# ==========================================================

def clean_blocks(
        blocks: List[DocumentBlock],
        config: CleanConfig = None
) -> Tuple[List[DocumentBlock], CleanReport]:
    """
    统一清洗入口
    :param blocks: 原始DocumentBlock列表（不会被修改）
    :param config: 清洗配置，不传则使用默认配置
    :return: (清洗后的新块列表, 清洗报告)
    """
    if config is None:
        config = CleanConfig()

    report = CleanReport(total_blocks=len(blocks))

    # 第一层：字符清洗
    char_cleaned = []
    for block in blocks:
        new_text = _normalize_characters(block.text, config, report)
        # 生成新对象，原对象完全不动
        char_cleaned.append(DocumentBlock(
            source_file=block.source_file,
            block_type=block.block_type,
            text=new_text,
            heading_path=block.heading_path.copy(),
            page=block.page,
            order=block.order
        ))

    # 第二层：布局清洗
    layout_cleaned = _clean_layout(char_cleaned, config, report)

    # 第三层：结构规整
    final_blocks = []
    for block in layout_cleaned:
        new_text = _normalize_structure(block.text, config)
        if new_text.strip():
            final_blocks.append(DocumentBlock(
                source_file=block.source_file,
                block_type=block.block_type,
                text=new_text,
                heading_path=block.heading_path.copy(),
                page=block.page,
                order=block.order
            ))

    return final_blocks, report


def print_report(report: CleanReport, filename: str = ""):
    """打印清洗报告，方便调试"""
    prefix = f"[{filename}] " if filename else ""
    print(f"{prefix}清洗完成：")
    print(f"  原始块数：{report.total_blocks}")
    print(f"  字符替换：{report.replaced_chars} 个")
    print(f"  删除Emoji：{report.removed_emojis} 个")
    print(f"  过滤页眉：{report.removed_headers} 行")
    print(f"  过滤页脚：{report.removed_footers} 行")
    print(f"  过滤空块：{report.removed_empty_blocks} 个")
    if report.warnings:
        print(f"  警告：{len(report.warnings)} 条")
        for w in report.warnings[:5]:
            print(f"    - {w}")
