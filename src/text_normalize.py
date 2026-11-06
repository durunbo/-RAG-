"""PDF 提取文本的字符归一：全项目唯一事实来源。

背景
----
pdftotext 从某些内嵌字体里取出来的不是汉字，而是「康熙部首 / CJK 部首补充」
区的部件字形。最典型的是「1 个月内」被取成「1 个⽉内」——`⽉` 是 U+2F49
（KANGXI RADICAL MOON），不是 U+6708 的「月」。

这两种字符肉眼几乎看不出差别，但会连锁出三个问题：
1. 逐字核验失败：模型输出正常汉字，原文存的是部首形，子串比对必然不中，
   表现为「同一个问题，一个答一个不答」；
2. embedding 偏移：向量模型看到的是生僻部件，不是常用汉字；
3. 关键词检索失效：用户搜「月」，库里是「⽉」。

处理原则
--------
1. 能用 NFKC 还原的一律还原。U+2F00–U+2FDF（Kangxi Radicals）100% 可还原，
   U+2E80–U+2EFF（CJK Radicals Supplement）只有一部分可还原；
2. NFKC 还原不了的，只对「对应汉字唯一」的字符手工映射（见 _RADICAL_FALLBACK）。
   偏旁形（如 ⺅ 单人旁、⺡ 三点水）本身不对应唯一汉字，一律不动，不猜；
3. **不做全局 NFKC**。全局 NFKC 会把 ①→1、℃→°C、㎏→kg、全角字母转半角，
   破坏编号层级和业务符号。这里只对部首区做定向归一。

本模块只依赖标准库：既被入库前的清洗（src/cleaner.py）使用，
也被回答生成时的核验（app/services/rag_service.py）使用。
"""

import re
import unicodedata
from typing import Iterator


# ==========================================================
# 需要定向归一的字符区间
# ==========================================================
_RADICAL_PATTERN = re.compile(r"[\u2e80-\u2eff\u2f00-\u2fdf\uf900-\ufaff]")

# NFKC 还原不了的部首字符 → 汉字。
# 收录标准：对应汉字唯一。偏旁形（⺅ ⺡ ⺣ ⻂ ⺘ …）不收，因为写成哪个字都可能错。
_RADICAL_FALLBACK = {
    # 简化字对应的部首形（字符名含 SIMPLIFIED，指向唯一汉字）
    "⻅": "见", "⻉": "贝", "⻋": "车", "⻓": "长", "⻔": "门",
    "⻙": "韦", "⻚": "页", "⻛": "风", "⻜": "飞", "⻢": "马",
    "⻥": "鱼", "⻦": "鸟", "⻧": "卤", "⻨": "麦", "⻩": "黄",
    "⻪": "黾", "⻬": "齐", "⻮": "齿", "⻰": "龙", "⻆": "角",
    # 繁体 / 异体写法，同样唯一
    "⻑": "长", "⻒": "长", "⻱": "龟",
    # 本身就是完整字形，不是偏旁
    "⺌": "小", "⺍": "小", "⺜": "日", "⺝": "月", "⺠": "民",
    "⺩": "王", "⺮": "竹", "⺧": "牛", "⺹": "老",
    "⻁": "虎", "⻄": "西", "⻣": "骨", "⻤": "鬼", "⻗": "雨",
    "⻘": "青",
}

_WHITESPACE = re.compile(r"\s+")

# 全角/半角形式区：NFKC 转成 ASCII 是标准行为，且不会碰到 ① ℃ ㎏
# （那些在 U+2460 / U+2100 / U+3300 区，不在此范围）
_HALFWIDTH_PATTERN = re.compile(r"[\uff00-\uffef]")

# 表意标点 → ASCII 标点：只做等价收敛，不删字符，所以不会削弱「逐字」校验。
# 注意：句末标点（。！？；）刻意不转换 —— rag_service 用它来切句、扩证据句，
# 一旦变成 ASCII 的 . 和 !，句子切分和「证据向前/向后补一句」会整片失效。
_PUNCT_TABLE = str.maketrans({
    "、": ",", "「": '"', "」": '"', "『": '"', "』": '"',
    "“": '"', "”": '"', "‘": "'", "’": "'", "《": "<", "》": ">",
    "〈": "<", "〉": ">", "【": "[", "】": "]", "〔": "(", "〕": ")",
    "—": "-", "–": "-", "·": ".", "•": ".",
})

# 变体选择符与零宽字符：纯格式修饰，不承载文字内容
_INVISIBLE_PATTERN = re.compile(r"[\ufe00-\ufe0f\u200b-\u200f\ufeff]")


def _restore_one(char: str) -> str:
    """单个部首字符 → 汉字；无法确定时原样返回。"""
    restored = unicodedata.normalize("NFKC", char)
    # NFKC 有时会把兼容字展开成多个字符，非空才算还原成功
    if restored and restored != char:
        return restored
    return _RADICAL_FALLBACK.get(char, char)


def restore_cjk_radicals(text: str) -> tuple[str, int]:
    """把部首区字符还原成对应汉字。

    :return: (还原后的文本, 实际替换的字符个数)
    """
    if not text:
        return text, 0

    replaced = 0

    def _sub(match: "re.Match[str]") -> str:
        nonlocal replaced
        original = match.group(0)
        restored = _restore_one(original)
        if restored != original:
            replaced += 1
        return restored

    return _RADICAL_PATTERN.sub(_sub, text), replaced


def iter_unmapped_radicals(text: str) -> Iterator[str]:
    """列出「既不能被 NFKC 还原、也不在确认表里」的部首字符。

    用于入库时告警：出现新字符说明确认表需要补一条，
    这正是「下次换个文档又踩同一个坑」的早期发现手段。
    """
    seen: set[str] = set()
    for char in text or "":
        if char in seen:
            continue
        if not _RADICAL_PATTERN.match(char):
            continue
        seen.add(char)
        if _restore_one(char) == char:
            yield char


def normalize_for_match(text: str) -> str:
    """核验用的严格归一：部首还原 → 全半角收敛 → 标点统一 → 去空白。

    核验问的是「模型输出的这几个字，原文里到底有没有」，
    比较双方必须用同一套归一。否则就会出现
    「原文没错、模型也没错、但就是比不中」的假拒答。

    除了康熙部首，PDF 还会把标点取成半角形（`､` U+FF64）或全角形，
    而模型输出的是常规标点；数字/字母也有全角半角之分。
    这些差异不承载语义，必须收敛掉，否则同样是随机拒答。
    """
    restored, _ = restore_cjk_radicals(text or "")
    restored = _INVISIBLE_PATTERN.sub("", restored)
    restored = _HALFWIDTH_PATTERN.sub(
        lambda m: unicodedata.normalize("NFKC", m.group(0)), restored
    )
    restored = restored.translate(_PUNCT_TABLE)
    return _WHITESPACE.sub("", restored)
