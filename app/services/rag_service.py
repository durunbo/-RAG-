"""RAG 服务：封装「改写查询 → 向量检索 → 构造 Prompt → 调用 LLM」的完整流程。

设计约束（刻意保持的边界）：
1. 不写数据库：会话与消息的持久化由 SessionManager 负责，本模块只读不写。
2. 不认识 HTTP：不抛 HTTPException，不 import app.schemas，便于脱离接口单测。
3. 只返回纯数据：sources 保持 VectorStore 的原始字典，schema 转换由路由层负责。

本模块曾经是「纯结构调整、行为与抽取前逐行一致」的状态（Step 1）。
后续批次在原流程上增加了三段策略，默认配置下行为不变：

- 低分过滤：min_score=0 时原样透传检索结果，不复制不过滤；
- 全量拒答：检索有结果但全部低于阈值时跳过 LLM，返回固定文案
  （知识库本身为空的情况仍走原生成路径，不算在内）；
- 来源标注：系统提示词要求输出 [文档N]，回答缺标注或编号越界只告警、不改写正文。

最新一批把「生成」换成了两阶段（详见下方「两阶段回答」注释块）：
模型只做受 JSON Schema 约束的条款抽取，答案由 Python 用核验过的记录拼装。
检索链路（Embedding / 切块 / 向量检索 / 重排 / min_score）完全未改动。
"""

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

from app.services.llm_client import LLMClient
from src.text_normalize import normalize_for_match
from src.vector_store.vector_store import VectorStore


logger = logging.getLogger(__name__)


# 来源标注格式：[文档1] / [文档 1]（容忍模型多打空格）
_CITATION_PATTERN = re.compile(r"\[文档\s*(\d+)\s*\]")


# 检索全部被低分过滤掉时的固定回答。
# 固定文案而不是让模型自由发挥：没有上下文时模型最容易编答案，
# 而这类问题恰恰是「知识库确实没有」的信号，应该让人一眼看出是拒答而非故障。
_NO_CONTEXT_ANSWER = "根据现有资料无法回答。"


# 追问识别：短句 + 含指代词/承接词，判定为「接着上文问」
# 例："那第二次呢？"、"这个怎么申请"、"还有呢"、"它扣多少"
_FOLLOWUP_MARKERS = (
    "那", "这", "它", "他", "她", "呢", "还", "再",
    "刚才", "上面", "上述", "前面",
)
_FOLLOWUP_MAX_LEN = 15  # 超过该长度视为自带完整语义，不做改写


# 并发控制：拿不到名额时最多等 100ms，之后快速失败
# 100ms 用于吸收瞬时抖动，又不至于让人明显感到在排队
_ACQUIRE_TIMEOUT_SECONDS = 0.1
_MIN_CONCURRENCY = 1
_MAX_CONCURRENCY = 10


# ==========================================================
# 两阶段回答：结构化抽取 → 确定性渲染
# ==========================================================
# 为什么不再用「一次自由生成」：
# 实测（A3/A5/A6）显示，自由生成时模型会先自造一条规定
# （例如「未经报备携带宠物进入公司是不允许的」），再在这条自造规定上推理，
# 于是去找「携带」的处罚、找不到，最后宣称「资料未明确说明处罚」——
# 而上下文里两条「饲养」罚则都在。Prompt 只能约束措辞，约束不住「先编一条
# 再在它上面推理」，因此这里把生成权收回来：
#   1) 模型只做抽取，输出受 JSON Schema 约束，每条记录必须带原文证据；
#   2) Python 逐条核验证据（必须是该片段的子串），核不上的记录直接丢弃；
#   3) 答案由核验过的记录拼装，模型没有机会再写入任何新事实。
# 代价是句式固定、不如自由生成自然 —— 刻意换取「每句话都能指回原文」。

# 抽取阶段的输出结构。字段刻意只说「这条记录讲什么」，不设
# 「非因公携带相关规定」这类空槽位：固定槽位会逼模型把甲条款塞进乙槽位
# （A3 实测 13/13 归属错误即由此而来）。
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["rule", "penalty"],
                    },
                    "action": {"type": "string"},
                    "condition": {"type": "string"},
                    "consequence": {"type": "string"},
                    "source_id": {"type": "integer"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "kind",
                    "action",
                    "condition",
                    "consequence",
                    "source_id",
                    "evidence",
                ],
            },
        },
        "gaps": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string"},
        },
    },
    "required": ["records", "gaps"],
}


_EXTRACTION_SYSTEM_PROMPT = (
    "你是企业制度条款抽取器。只做抽取，不回答用户问题，不做推理，不做总结。\n"
    "\n"
    "抽取规则：\n"
    "1. 只抽取与用户所问对象直接相关的条款。判断标准是行为对象相同：\n"
    "   用户问宠物，就只抽针对宠物的条款；着装、资产、停车、卫生、快递等\n"
    "   其它主题一律不抽，也不要为它们生成记录。\n"
    "2. 逐个片段检查。上下文里有几个 [文档N]，就要分别看一遍，\n"
    "   每个片段里出现相关条款都要抽出来，不要只抽第一个片段。\n"
    "   不要因为已经从某个片段抽到了相关内容就跳过后续片段。\n"
    "   同一主题下如果多个片段都在讲这个主题的不同方面\n"
    "   （比如一个讲入职流程、一个讲入职要交的材料），\n"
    "   每个片段里的具体内容都要分别抽出。\n"
    "   判断哪个更切题是上层的事，你的任务是宁多勿漏；\n"
    "   跨主题的（用户问宠物就不抽着装）仍然不抽。\n"
    "   同一个行为可能在多份制度里都有规定（一份写乐捐、一份写扣分），\n"
    "   必须分别抽出，各写一条。漏掉某个片段里明确写着的处罚是最严重的错误。\n"
    "   原文用冒号、分号串起多项内容时（材料清单、禁止事项、扣分档位等），\n"
    "   必须把整串具体内容覆盖到，不能只抽列举前后的总结句。\n"
    "   判定方法：如果你抽出的 evidence 只有「准备好了就可以…」「以上材料…」\n"
    "   「按上述要求执行」这类收尾话，说明你抽错了，用户拿不到任何一条具体内容。\n"
    "   正确做法是把冒号后面那一整串逐字抄下来（可拆成多条，每项各一条）。\n"
    "   同样地，action 不能写成「准备材料」「办理手续」「走流程」这类概括词，\n"
    "   必须是原文里连续存在的那段话。\n"
    "3. 一条记录对应原文中的一项规定。用分号、顿号串起来的列举算同一项规定，\n"
    "   此时可以整段抄录，不受句数限制；列举之外的其它规定仍要分开成多条记录。\n"
    "   同一片段里有几句互不相同的规定，就写几条记录，不要合成一条长记录。\n"
    "   下面每个字段都会被程序逐字回查原文，对不上的记录整条丢弃，\n"
    "   所以：不要概括，不要改写，不要把自己的话填进字段。\n"
    "   - action：原文对该行为的完整说法，逐字截取，包含范围、对象、限定词。\n"
    "     原文写「严禁在公司范围内饲养任何宠物，鱼类除外」，\n"
    "     就写「在公司范围内饲养任何宠物」，不要只写「饲养」；\n"
    "     原文写「因公需求携带宠物进入公司的」，就写「因公需求携带宠物进入公司」，\n"
    "     不要缩成「携带」，更不要写成原文没有的说法（如「未经报备携带」）。\n"
    "   - condition：原文对这一行为的适用限定（如「鱼类除外」「寝室内」）\n"
    "     或必须履行的要求（如「需提前一天向行政中心提交报备申请」）；\n"
    "     这些信息在原文里有，就必须写出来，不要留「无」；确实没有才写「无」。\n"
    "   - consequence：原文对这一行为规定的处罚或后果；原文没有处罚就写「无」。\n"
    "     证据句里出现「乐捐」「扣除…分」「赔偿」「追究责任」这类处罚用语时，\n"
    "     必须把这个处罚写进 consequence，不能填「无」。\n"
    "   - kind：原文对该行为规定了处罚（乐捐、扣分、赔偿、追究责任等）才填 penalty；\n"
    "     报备、登记、审批、接待这类流程要求属于 rule，此时 consequence 填「无」。\n"
    "   - source_id：该句所在片段序号 N（上下文里的 [文档N]），必须是整数。\n"
    "   - evidence：该句原文，逐字照抄，不改写、不拼接、不省略关键词。\n"
    "4. 一个片段里含多个不同行为时，拆成多条记录。\n"
    "5. 不要补充原文没有的触发条件。原文只写「饲养」，就不要写成「未经报备饲养」；\n"
    "   原文没有提到「非因公」，就不要提「非因公」。\n"
    "6. 不要把不同行为的处罚合并，也不要把甲行为的处罚写到乙行为上。\n"
    "7. gaps 只用来登记「用户问题里提到的情形，在原文中完全没有规定」：\n"
    "   写一句「X 是否允许」这样的说法，X 必须是用户问题里的说法。\n"
    "   用户问「能不能/可不可以」而原文只规定了其中一种情形时（例如只写了\n"
    "   因公需求携带），把用户问的另一种情形登记为 gap（如「非因公携带宠物是否允许」）。\n"
    "   这里只能说明原文没写，不许写任何规定、金额、分值、流程、部门或判断。\n"
    "   所有片段已覆盖问题，或问题与片段完全无关，就返回空数组。\n"
    "8. records 为空表示这批片段答不了这个问题；这时 gaps 也返回空数组，\n"
    "   不要用其它主题的条款凑数。\n"
    "\n"
    "只输出 JSON，不要输出任何解释文字。\n"
)

# 抽取阶段的输出上限。JSON + 原文证据比自然语言回答长得多，
# 沿用回答阶段的 400 token 会在第 4~5 条记录处被截断，
# 半截 JSON 解析失败就等于把这个本可回答的问题降级成拒答。
_EXTRACTION_NUM_PREDICT = 1200

# ----------------------------------------------------------
# 按需改写：只在「抽取空手」时把用户口语问题对齐到库内问法
# ----------------------------------------------------------
# 为什么必须带参考问法：让模型自由改写会写成「入职流程是什么」这种更抽象的
# 说法，抽取阶段照样判定「对象不同」而返回空。带上检索到的库内问法，
# 改写才能落到原文真正在讲的那件事上（实测「面试过了之后要干啥」
# 自由改写后仍拒答，对齐改写后直接抽到完整材料清单）。
# 为什么不设 same_topic 字段：7B 模型做「先判定、再决定要不要改写」的两步推理
# 不稳定。实测 3 条本该改写的问题（面试过了要干啥 / 上班穿什么 / 想辞职走什么流程）
# 全被判成 false，rewritten 直接返回空串，改写率 0/3；去掉该字段、让它直接输出
# 改写句或原样输出问题后，同一批 2/3 改写成功，33 条实测里救回 3 条、0 次误改。
# 判断「是否需要改写」这件事，交给下面的显式条件（非空 / 长度 / 与原问题有共同片段）。
_ALIGN_SYSTEM_PROMPT = (
    "你是企业问答系统的问句对齐器。给你一个用户问题和若干条库内已有的参考问法。\n"
    "任务：如果候选里有与用户问题是同一件事的那一条，就把用户问题改写成它问的那件事；\n"
    "如果候选里没有任何一条是同一件事，就把用户问题原样输出。\n"
    "规则：\n"
    "1. 判断依据是「核心对象和场景是否一致」，措辞不同不算不同的事。\n"
    "   用户说「还有啥安排」「要做什么」「需要办什么」这类笼统说法时，\n"
    "   只要候选讲的是同一件事，就算同一件事。\n"
    "   候选有多条时只挑最接近的一条，不要综合多条。\n"
    "2. 需要改写时，改写句要带出挑中那条的核心名词（如「材料」「离职」「转正」），\n"
    "   不能写成「有哪些安排」「要做什么」这类抽象说法；不超过 25 个字，只写问句本身。\n"
    "   示例：\n"
    "     用户问题：面试通过了后面还有啥安排\n"
    "     候选：面试通过啦，我需要准备什么材料来办岗前培训呀？\n"
    "     -> 正确改写：面试通过后需要准备什么材料\n"
    "3. 不需要改写时，rewritten 必须原样等于用户问题，不要返回空字符串。\n"
    "只输出 JSON。\n"
)

_ALIGN_SCHEMA = {
    "type": "object",
    "properties": {"rewritten": {"type": "string"}},
    "required": ["rewritten"],
}

_ALIGN_NUM_PREDICT = 120
_ALIGN_MAX_CHARS = 30
# 参考问法取不到时的兜底片段长度（非 FAQ 文档没有 question 字段，用正文开头）
_ALIGN_REFERENCE_CHARS = 40
# 交给模型挑选的候选问法条数。只看 Top-1 会踩坑：「面试完了该干嘛」的 Top-1
# 是离职块（0.6270），而正确块排第 2（0.6108），只给一条等于逼模型判否。
_ALIGN_REFERENCE_COUNT = 3

# 空白归一等字符级处理统一收在 src/text_normalize.py：
# PDF 抽出的正文里有换行、以及「乐捐200 元/次」这种夹空格的写法，
# 逐字比对前必须先归一，否则「照抄原文」的证据会被误判成核验失败。
_LINE_BREAK = re.compile(r"[ \t]*\n[ \t]*")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# 「后果」是否真的是处罚。用来挡住把「需提前一天报备」这类
# 流程要求当成处罚渲染到「相关处罚」段落里。
_PENALTY_SIGNAL = re.compile(
    r"乐捐|罚款|处罚|扣|赔偿|警告|通报|追究|处分|辞退|开除|黑名单|行政分|承担"
)

# gaps 里出现这些词说明模型又在写规定了 —— 直接丢弃。
# 「允许」不在此列：「是否允许」正是登记「原文没写」的正常句式。
_GAP_FORBIDDEN = re.compile(
    r"[0-9０-９]|乐捐|罚款|扣分|行政分|处分|辞退|开除|严禁|禁止|不得|必须|应当|一律"
)

# 原文把适用前提限定为「因公」时，用它把限定词取反，得到原文没写的那种情形。
_PERMIT_RESTRICT = re.compile(r"因公(?:需求|需要)?")
# 只有「能不能做」这类许可型问句才推导「另一种情形是否允许」：
# 问「有什么处罚」时补一句「是否允许」属于答非所问。
_PERMIT_QUESTION = re.compile(r"可以|能否|能不能|可否|是否允许|允许|行不行")
# 取反后剥掉这些尾巴，避免出现「非因公携带宠物的情况除外是否允许」这种句子
_GAP_TAIL = re.compile(r"(?:的情况?)?(?:除外|之外)?$")

# 词面重合兜底：问题里的这些说法过于通用，参与比对会让任何片段都算「相关」
_QUESTION_NOISE_PHRASES = (
    "请问", "一下", "可不可以", "可以", "能否", "怎么办", "怎么样", "怎么",
    "什么", "哪些", "多少", "如何", "有没有", "是否", "我们", "你们",
    "公司", "员工", "上班", "工作期间", "这个", "那个",
)
_QUESTION_STOP_CHARS = frozenset("的了吗呢啊吧呀是我你他她它和与或在有就该都也还再")

# 问题口语说法 → 原文用词。只作用于闸门的字面比对，不进入答案。
# 收录标准：同一件事、原文不会照这个说法写、且漏了就会整批拒答。
_QUERY_SYNONYMS = (
    ("衣服", "穿着"),
    ("穿啥", "穿着"),
)

_CN_NUMERALS = "一二三四五"



# ==========================================================
# 配置与输出结构
# ==========================================================
@dataclass
class RAGConfig:
    """RAG 流程参数。默认值与抽取前 router 中的常量保持一致。"""

    max_sources: int = 5     # 最多使用的文档片段数量
    context_window: int = 3  # 滑动上下文窗口大小
    max_concurrency: int = 3 # 同时进行的问答上限（单卡 GPU 实际串行）
    min_score: float = 0.0   # 检索片段的最低相似度，0 表示不过滤
    enable_align_rewrite: bool = True  # 抽取空手时是否按需改写问题重试一次


@dataclass
class RAGResult:
    """RAG 流程输出。

    sources 为 VectorStore 返回的原始字典列表（含 score / text / metadata），
    由调用方决定如何转换为具体的 API schema。
    """

    answer: str
    sources: list[dict]
    retrieval_ms: float = 0.0   # 检索阶段耗时，用于运行指标
    generation_ms: float = 0.0  # 生成阶段耗时，用于运行指标


# ==========================================================
# 异常
# ==========================================================
class RAGServiceError(Exception):
    """RAG 流程异常基类。"""


class RetrievalUnavailable(RAGServiceError):
    """向量检索不可用（数据库异常、连接失败等）。"""


class GenerationFailed(RAGServiceError):
    """LLM 生成失败（模型不可用、超时、响应格式异常等）。"""


class ServiceBusy(RAGServiceError):
    """并发已满，快速失败而非排队。"""


# ==========================================================
# 抽取结果校验（纯函数，便于单测）
# ==========================================================
def _normalize(text: str) -> str:
    """核验用的严格归一：还原 PDF 部首字符 + 去掉全部空白。

    必须与入库清洗共用同一套归一（src/text_normalize）。此前这里只去空白，
    而库里的「1 个⽉内」是康熙部首 U+2F49，模型输出正常汉字「月」，
    两边永远比不中，表现就是「同一个问题，一个答一个不答」。
    """
    return normalize_for_match(text or "")


def _is_grounded(value: str, normalized_evidence: str, normalized_chunk: str) -> bool:
    """判断一个字段是否能在这条记录的证据句（或它所属片段）里找到。

    这是整套方案的安全阀：核不上的内容一律不展示，
    模型想补一句「未经报备」也补不进去——原文没这几个字。
    """
    normalized = _normalize(value)
    if not normalized or normalized.lower() in ("无", "none", "null", "n/a"):
        return False
    return normalized in normalized_evidence or normalized in normalized_chunk


def _is_subsequence(needle: str, haystack: str) -> bool:
    """needle 的字符能否按顺序在 haystack 中依次找到。

    比「子串」宽松，但仍要求每个字、每个词的先后顺序都出自原文：
    模型把「饲养任何宠物」写成「饲养宠物」时可以通过，
    而把原文没有的「未经报备」塞进来则一定过不了 —— 那四个字原文里没有。
    """
    return _align_span(needle, haystack) is not None


def _align_span(needle: str, haystack: str) -> Optional[tuple[int, int]]:
    """按字序把 needle 对到 haystack 上，返回首末位置；对不上返回 None。"""
    if not needle:
        return None

    position = 0
    start = -1
    end = -1
    for char in needle:
        found = haystack.find(char, position)
        if found < 0:
            return None
        if start < 0:
            start = found
        end = found
        position = found + 1

    return start, end


def _repair_action(action: str, normalized_evidence: str) -> Optional[str]:
    """把「字序对得上、但不是连续原文」的行为名补成连续原文。

    例：模型写「在公司范围内饲养宠物」，原文是「在公司范围内饲养任何宠物」，
    补齐后是后者 —— 渲染出来才是能逐字回查的引文，而不是省略版的转述。
    扩张幅度设上限，避免把整句都吞进来。
    """
    if not action or action in normalized_evidence:
        return None

    span = _align_span(action, normalized_evidence)
    if span is None:
        return None

    start, end = span
    candidate = normalized_evidence[start:end + 1]
    if len(candidate) > len(action) * 1.5 + 6:
        return None

    return candidate


def _parse_extraction(raw: str) -> dict:
    """解析抽取结果。失败抛 ValueError，由调用方降级为拒答。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("抽取结果为空")

    text = _JSON_FENCE.sub("", raw.strip())

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"抽取结果不是合法 JSON：{exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("抽取结果顶层不是对象")

    return data


def _merge_duplicate_records(records: list[dict]) -> list[dict]:
    """同一规定被检索成多个片段时（如两份制度重复同一句话），合并为一条并保留多个出处。"""
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []

    for record in records:
        key = (
            record["kind"],
            _normalize(record["evidence"]),
            _normalize(record["consequence"] or ""),
        )
        if key in merged:
            if record["source_id"] not in merged[key]["sources"]:
                merged[key]["sources"].append(record["source_id"])
            continue

        item = dict(record)
        item["sources"] = [record["source_id"]]
        merged[key] = item
        order.append(key)

    # 同一句话在不同制度里重复出现时，模型可能一边抽成一句、另一边抽成整段。
    # 这里按「证据互相包含」再合一次：保留更短更精确的那条，把出处并进去，
    # 避免同一条规定在答案里出现两遍（长的那条还会拖长答案）。
    result: list[dict] = []
    for item in (merged[key] for key in order):
        normalized_item = _normalize(item["evidence"])
        item_consequence = _normalize(item["consequence"] or "")

        for kept in result:
            normalized_kept = _normalize(kept["evidence"])
            if not (
                normalized_kept == normalized_item
                or normalized_kept in normalized_item
                or normalized_item in normalized_kept
            ):
                continue

            same_kind = kept["kind"] == item["kind"]
            if same_kind:
                # 后果要么相同，要么一句包含另一句（「乐捐200元/次」与
                # 「乐捐200元/次，并立即将宠物带离公司」是同一条处罚的两种截取长度）
                kept_consequence = _normalize(kept["consequence"] or "")
                if not (
                    kept_consequence == item_consequence
                    or kept_consequence in item_consequence
                    or item_consequence in kept_consequence
                ):
                    continue

            for source in item["sources"]:
                if source not in kept["sources"]:
                    kept["sources"].append(source)

            if same_kind:
                # 保留信息更全的那份证据与后果，避免同一规定出现两遍
                if len(normalized_item) < len(normalized_kept):
                    kept["evidence"] = item["evidence"]
                    kept["action"] = item["action"]
                if len(item_consequence) > len(_normalize(kept["consequence"] or "")):
                    kept["consequence"] = item["consequence"]
            elif item["kind"] == "penalty":
                # 同一句话被一边抽成规范、一边抽成处罚：保留处罚（信息更全，
                # 且它有据可查的后果），否则处罚会挂在「规定明确」段落里
                kept["kind"] = "penalty"
                kept["action"] = item["action"]
                kept["consequence"] = item["consequence"]
                kept["condition"] = item["condition"] or kept["condition"]
                kept["evidence"] = item["evidence"]
            break
        else:
            result.append(item)

    return result


def _validate_records(
    raw_records: object,
    chunk_texts: dict[int, str],
) -> list[dict]:
    """逐条核验抽取结果，核不上的直接丢弃。

    Args:
        chunk_texts: {片段编号: 片段原文}，编号必须与提示词里的 [文档N] 一致。

    丢弃条件（任一条不满足即丢）：
    1. source_id 必须是本次上下文里出现过的编号；
    2. evidence 必须是该片段的逐字子串（去空白后比对）；
    3. action 必须能在这条证据（或它的前后各一句）里找到；
    4. 判成处罚时，consequence 必须有据，且行为名与后果在同一句/相邻两句里。

    注意第 4 条不是「丢弃」而是「降级」：后果无据、或只在片段的远处句子出现时，
    这条记录按规范渲染（直接引用原文证据），不产出「行为 → 处罚」的对应关系。
    这样既不会把真处罚判死，也不会把甲行为的处罚安到乙行为头上。
    """
    if not isinstance(raw_records, list):
        return []

    normalized_chunks = {index: _normalize(text) for index, text in chunk_texts.items()}
    validated: list[dict] = []

    for index, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            logger.warning("丢弃第 %s 条抽取记录：不是对象", index)
            continue

        try:
            source_id = int(raw.get("source_id"))
        except (TypeError, ValueError):
            logger.warning("丢弃第 %s 条抽取记录：source_id 非法 %r", index, raw.get("source_id"))
            continue

        if source_id not in normalized_chunks:
            logger.warning(
                "丢弃第 %s 条抽取记录：source_id=%s 不在本次上下文 %s 内",
                index,
                source_id,
                sorted(normalized_chunks),
            )
            continue

        evidence = str(raw.get("evidence") or "").strip()
        normalized_evidence = _normalize(evidence)
        if len(normalized_evidence) < 6:
            logger.warning("丢弃第 %s 条抽取记录：证据过短 %r", index, evidence[:40])
            continue

        chunk = normalized_chunks[source_id]
        if normalized_evidence not in chunk:
            logger.warning(
                "丢弃第 %s 条抽取记录：证据在 [文档%s] 中找不到 %r",
                index,
                source_id,
                evidence[:60],
            )
            continue

        action = str(raw.get("action") or "").strip()
        normalized_action = _normalize(action)
        if not normalized_action:
            logger.warning("丢弃第 %s 条抽取记录：行为名为空", index)
            continue

        action_in_evidence = (
            normalized_action in normalized_evidence
            or _is_subsequence(normalized_action, normalized_evidence)
        )

        if not action_in_evidence:
            # 行为名落在证据的上一句：模型把「严禁…」与「若发现…乐捐」拆开了，
            # 向前补一句即可确认对应关系，仍要求补完是原文连续片段
            prepended = _prepend_one_sentence(normalized_evidence, chunk)
            if prepended and (
                normalized_action in prepended
                or _is_subsequence(normalized_action, prepended)
            ):
                logger.info(
                    "第 %s 条记录：行为名在证据的上一句里，证据向前补一句", index
                )
                evidence = prepended
                normalized_evidence = prepended
                action_in_evidence = True

        if not action_in_evidence:
            if (
                normalized_action not in chunk
                and not _is_subsequence(normalized_action, chunk)
            ):
                logger.warning(
                    "丢弃第 %s 条抽取记录：行为名在原文中找不到 %r", index, action[:60]
                )
                continue
            # 行为名落在同一片段的其它句子里：仍属原文用词，规范类可用，留痕
            logger.warning(
                "第 %s 条记录的行为名不在其证据句内、但属于同一片段：%r",
                index,
                action[:60],
            )
        elif normalized_action not in normalized_evidence:
            # 字序对得上但不是连续原文：补成连续原文，保证渲染出来能逐字回查
            repaired = _repair_action(normalized_action, normalized_evidence)
            if repaired:
                logger.info("行为名补成原文连续片段：%r -> %r", action[:40], repaired[:40])
                action = repaired
                normalized_action = repaired

        condition = str(raw.get("condition") or "").strip()
        consequence = str(raw.get("consequence") or "").strip()
        declared_kind = str(raw.get("kind") or "").strip()

        condition_grounded = _is_grounded(condition, normalized_evidence, chunk)

        # 处罚的后果必须落在这条证据句里，不接受「同一片段的别的句子」：
        # 否则会出现「行为取自 A 句、后果取自 B 句」的跨句错配
        # （实测出现过把「饲养宠物」配上「鱼缸引发火灾追究法律责任」），
        # 而两半单独看都能在原文找到，只做片段级校验发现不了。
        normalized_consequence = _normalize(consequence)
        consequence_grounded = (
            bool(normalized_consequence)
            and normalized_consequence not in ("无", "none", "null", "n/a")
            and normalized_consequence in normalized_evidence
        )

        if normalized_consequence and not consequence_grounded:
            # 模型把一条规定的两句拆成两条证据时，向后补一句再看一次：
            # 相邻两句本来就是同一条规定，接回来才不会把真处罚判成无据。
            extended = _extend_one_sentence(normalized_evidence, chunk)
            if extended and normalized_consequence in extended:
                logger.info(
                    "第 %s 条记录：后果在证据的下一句里，证据向后扩一句", index
                )
                evidence = extended
                normalized_evidence = extended
                consequence_grounded = True

        # kind 不采信模型的声明，直接由「后果是否有据、是否像处罚」推导：
        # 实测模型常把「必须报备」标成 penalty、把「扣除行政分1分」标成 rule，
        # 标签错了会把流程要求挂进处罚段、或把真处罚降级成规范。
        # 数据能判的就交给数据判 —— 这也正是这个方案要的：少一处自由判断。
        is_penalty = bool(consequence_grounded and _PENALTY_SIGNAL.search(consequence))

        if is_penalty and not action_in_evidence:
            logger.warning(
                "丢弃第 %s 条处罚记录：行为名与后果不在同一句，无法确认对应关系 %r",
                index,
                action[:60],
            )
            continue

        if declared_kind and (declared_kind == "penalty") != is_penalty:
            logger.info(
                "第 %s 条记录的 kind=%s 与后果不符，按后果判定为 %s：%r",
                index,
                declared_kind,
                "penalty" if is_penalty else "rule",
                consequence[:40],
            )

        if is_penalty:
            condition_out = condition if condition_grounded else ""
            consequence_out: Optional[str] = consequence
        else:
            # 规范类记录没有「后果」：若模型把报备/流程要求填到了 consequence，
            # 就把它当作适用条件展示，避免丢信息，也避免挂进处罚段。
            if condition_grounded:
                condition_out = condition
            elif consequence_grounded:
                condition_out = consequence
            else:
                condition_out = ""
            consequence_out = None

        validated.append(
            {
                "kind": "penalty" if is_penalty else "rule",
                "action": action,
                "condition": condition_out,
                "consequence": consequence_out,
                "source_id": source_id,
                # 规范类记录直接引用原文证据，不依赖字段填得准不准。
                # 收尾句（「以上材料…」）在这里补成自足的枚举段：
                # 放在所有校验之后做，避免放宽「后果必须同句」的约束。
                "evidence": _clean_evidence(
                    evidence
                    if is_penalty
                    else _expand_referential_evidence(
                        evidence, chunk_texts.get(source_id, "")
                    )
                ),
            }
        )

    return _merge_duplicate_records(validated)


# gap 的疑问尾巴（「…是否允许」）：只标记句式，本身不含信息，比较时先剥掉
_GAP_QUESTION_TAIL = re.compile(r"(?:是否|能否|可否|可不可以|允不允许).*$")


def _is_gap_paraphrase(question: str, gap: str) -> bool:
    """判断这条「原文未明确」是不是只是把用户的问题抄了一遍。

    实测问题：问「工资每个月几号发」，答案上文已经给出「工资每月 15 号准时到账」，
    下文却登记「现有资料未明确说明：工资每个月几号发」。等于自己打自己的脸。
    同样出现在「上班迟到扣多少钱」这类问法上。

    根因是模型把问题原样填进了 gaps，而 _validate_gaps 只校验长度和禁用词，
    看不出「这句话没有提供任何新信息」。
    """
    # 只比「是否允许」前面那半句：疑问尾巴本身不含信息，
    # 带着它一起比会把覆盖率摊薄，反而放过复述句。
    core = _GAP_QUESTION_TAIL.sub("", gap).strip()
    if len(core) < 3:
        return False

    question_grams = _query_bigrams(question)
    core_grams = _query_bigrams(core)
    if not question_grams or not core_grams:
        return False

    overlap = len(core_grams & question_grams) / len(core_grams)
    return overlap >= 0.6


def _gap_off_topic(question: str, gap: str) -> bool:
    """gap 说的对象跟用户问的根本不是一回事。

    实测问题：问「宿舍里可以养鱼吗」，登记「非因公携带宠物是否允许」——
    用户问的是鱼，这条讲的是宠物，纯属答非所问，还占掉了唯一一条「未明确」。

    判据用单字交集而不是二字片段：模型填 gap 时倾向照抄原文用词（「宠物」），
    用户提问用口语（「狗」），二字片段必然对不上，但单字「带」还能救回来。
    只有连一个实字都不重合，才算跑题。
    """
    question_chars = _query_chars(question)
    gap_chars = _query_chars(_GAP_QUESTION_TAIL.sub("", gap) or gap)
    if not question_chars or not gap_chars:
        return False
    return not (question_chars & gap_chars)


def _gap_contradicts_records(gap: str, records: Optional[list[dict]]) -> bool:
    """判断这条「未明确」是不是在说原文其实已经写明的事。

    实测问题：问「工资几号发」，答案已经给出「工资每月 15 号准时到账」，
    却登记「现有资料未明确说明：每月发工资的具体日期是否规定」；
    问「工牌丢了怎么补办」，答了「可前往 406 室办理」，
    却登记「工牌丢失是否允许补办」。

    判据：gap 是「是否允许 / 是否规定」这类句式，且它提到的对象在已核验的
    记录里出现过两个及以上二字片段 —— 说明原文写过这件事，不该说"没写明"。
    阈值取 2 而不是 1：只重合一个片段（例如都提到"工牌"）时，
    gap 问的可能是另一个维度的信息（能不能不戴 vs 怎么补办），留着无害。
    """
    if not records:
        return False
    if not re.search(r"是否允许|是否规定|是否明确|是否说明|有无规定", gap):
        return False

    gap_grams = _query_bigrams(gap)
    if not gap_grams:
        return False

    covered: set[str] = set()
    for record in records:
        covered |= _query_bigrams(
            " ".join(
                filter(
                    None,
                    [
                        record.get("action") or "",
                        record.get("condition") or "",
                        record.get("evidence") or "",
                    ],
                )
            )
        )

    return len(gap_grams & covered) >= 2


def _validate_gaps(
    raw_gaps: object,
    question: str = "",
    records: Optional[list[dict]] = None,
) -> list[str]:
    """核验「原文未明确」的登记项：只允许存在「某某是否允许」这类说法。"""
    if not isinstance(raw_gaps, list):
        return []

    accepted: list[str] = []
    for raw in raw_gaps:
        if not isinstance(raw, str):
            continue

        text = raw.strip()
        for prefix in (
            "现有资料未明确说明：",
            "现有资料未明确说明:",
            "原文未明确说明：",
            "原文未明确说明:",
            "原文未明确：",
            "原文未明确:",
        ):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        if not text or len(text) > 60:
            continue
        if _GAP_FORBIDDEN.search(text):
            logger.warning("丢弃无依据的「未明确」陈述（含规定性用语）：%r", text[:60])
            continue
        if question and _is_gap_paraphrase(question, text):
            logger.info("丢弃只是复述问题的「未明确」登记：%r", text[:60])
            continue
        if question and _gap_off_topic(question, text):
            logger.info("丢弃与问题无关的「未明确」登记：%r", text[:60])
            continue
        if _gap_contradicts_records(text, records):
            logger.info("丢弃与已核验记录矛盾的「未明确」登记：%r", text[:60])
            continue

        accepted.append(text)

    return accepted[:3]


def _derived_gaps(question: str, records: list[dict]) -> list[str]:
    """推导「原文只规定了因公情形，用户问的范围还包括非因公」。

    只在一种情况下产出一条：原文条款把前提限定为「因公」，而用户问题本身
    没提「因公」—— 说明用户问的范围比原文宽，非因公那一半原文没写。
    这时登记「非因公……是否允许」，绝不替用户判定允许或禁止。

    为什么由代码算而不让模型填：同一个问题实测 5 轮里只有 2 轮填了 gap，
    而这条恰恰最容易让用户误解 —— 不写，他可能以为问的情形已被回答。
    这里只做「把原文的限定词取反」这一件事，不引入任何新事实。
    """
    if "因公" in (question or ""):
        return []
    # 问「有什么处罚」时不需要「非因公是否允许」这一条
    if not _PERMIT_QUESTION.search(question or ""):
        return []

    question_grams = _query_bigrams(question)
    for record in records:
        action = record["action"]
        if "因公" not in action and "因公" not in (record["condition"] or ""):
            continue
        # 用户问的必须就是这条记录说的事。实测问「宿舍里可以养鱼吗」时，
        # 因为片段里另有「因公需求携带宠物…」，被推出「非因公携带宠物是否允许」，
        # 答非所问。这里要求问题与该记录的行为名有词面交集。
        if question_grams and not any(g in _normalize(action) for g in question_grams):
            continue

        phrase = _GAP_TAIL.sub("", _strip_tail(_PERMIT_RESTRICT.sub("非因公", action)))
        if len(_normalize(phrase)) < 4:
            continue
        return [f"{phrase}是否允许"]

    return []


def _merge_gaps(model_gaps: list[str], derived_gaps: list[str]) -> list[str]:
    """合并模型登记的 gap 与代码推导的 gap：同一件事只留一条。"""
    if not derived_gaps:
        return model_gaps
    # 模型自己已经写了「非因公…」，就不再重复追加
    if any("非因公" in gap for gap in model_gaps):
        return model_gaps
    return (derived_gaps + model_gaps)[:3]


def _query_kept(question: str) -> str:
    """按闸门口径清洗问题：同义归一 + 去噪音短语 + 去停用字。

    同义归一：只统一「比对用的字面」，不改动答案里的任何文字。
    「上班要穿什么衣服」剥掉噪音词后剩「要穿衣服」，二字片段是
    「要穿/穿衣/衣服」，而原文写的是「需穿着公司文化衫或西服」，
    零重合会让闸门把正确答案整批拒掉。
    """
    text = question or ""
    for phrase, canonical in _QUERY_SYNONYMS:
        text = text.replace(phrase, canonical)
    for phrase in _QUESTION_NOISE_PHRASES:
        text = text.replace(phrase, " ")

    return "".join(
        char
        for char in text
        if (char.isalnum() or "\u4e00" <= char <= "\u9fff")
        and char not in _QUESTION_STOP_CHARS
    )


def _query_chars(question: str) -> set[str]:
    """实字集合，用于判断两句话说的是不是同一个对象（比二字片段宽松）。"""
    return set(_query_kept(question))


def _query_bigrams(question: str) -> set[str]:
    """把问题拆成二字片段，用于判断抽取结果是否与问题沾边。"""
    kept = _query_kept(question)
    if len(kept) < 2:
        return set()

    return {kept[i:i + 2] for i in range(len(kept) - 1)}


def _topically_related(
    question: str,
    records: list[dict],
    chunk_texts: Optional[dict[int, str]] = None,
) -> bool:
    """兜底：记录与问题连一个二字片段都不重合，就当作与本问题无关。

    这是「不用无关制度拼答案」的第二道闸。问题与原文用词完全同义但不同字时
    （例如问「宠物」原文写「犬只」）会误判为无关 —— 宁可拒答，
    也不要把别的制度拼进来。

    比对范围包含「该记录引用的原文块」而不只是抽取出来的那几个字段。
    原因：模型抄 evidence 时只会抄答案那几句，块首的问题行抄不进来。
    FAQ 类切块的 Q 行恰恰承载了用户问法的原词（「面试通过啦，我需要
    准备什么材料…」），只看 evidence 就会把「面试通过了后面要做什么」
    误判成无关而拒答。evidence 本身已经过逐字核验、记录也认了引用编号，
    所以块原文与记录同源，把它纳入比对不引入无关内容。
    """
    grams = _query_bigrams(question)
    if not grams:
        return True

    for record in records:
        parts: list[str] = [
            record["action"],
            record["condition"],
            record["consequence"],
            record.get("evidence") or "",
        ]

        if chunk_texts:
            for source_index in record.get("sources") or ():
                cited_text = chunk_texts.get(source_index)
                if cited_text:
                    parts.append(cited_text)

        haystack = _normalize(" ".join(filter(None, parts)))
        if any(gram in haystack for gram in grams):
            return True

    return False


_SENTENCE_END = "。；;！？?"
_SENTENCE_SPLIT = re.compile(r"[。；;！!？?\n]+")


def _extend_one_sentence(evidence: str, chunk: str) -> Optional[str]:
    """把证据向后扩一句（仍是原文连续片段）。

    用途：模型常把「严禁…饲养任何宠物，鱼类除外。」和紧跟的
    「若发现违规饲养宠物，饲养者将乐捐200 元/次…」拆成两条证据，
    导致处罚句不在证据里、整条处罚被判为无据。前后相邻的两句在原文里
    本来就是一条完整规定，扩一句能把它们接回来；再远就不扩了 ——
    实测把不相邻的句子连起来正是「行为取 A 句、后果取 B 句」错配的来源。
    """
    start = chunk.find(evidence)
    if start < 0:
        return None

    end = start + len(evidence)
    for index in range(end, len(chunk)):
        if chunk[index] in _SENTENCE_END:
            return chunk[start:index + 1]

    return None


def _prepend_one_sentence(evidence: str, chunk: str) -> Optional[str]:
    """把证据向前扩一句（仍是原文连续片段）。

    与向后扩成对：模型也会把行为名写成上句的「严禁…饲养任何宠物」，
    却把证据只截成下句「若发现…乐捐200 元/次」，于是「行为名不在证据里」
    被判无据。前后相邻两句本是同一条规定，接回来即可；再远不扩。
    """
    start = chunk.find(evidence)
    if start <= 0:
        return None

    # 从 start-2 往回找：start-1 通常正是证据前一句的句末标点，
    # 要的都是再往前那一个，否则「补一句」变成「原地不动」
    cut = -1
    for index in range(start - 2, -1, -1):
        if chunk[index] in _SENTENCE_END:
            cut = index
            break

    return chunk[cut + 1:start + len(evidence)]


# 列举段的边界只认句末标点，**刻意不含分号**：分号本身是列举项的分隔符，
# 按分号切会把「整串材料」切成一条条，正好丢掉了要补的那部分。
_CLAUSE_SPLIT = re.compile(r"(?<=[。！？?!])")
# 「以上材料」「上述情形」这类收尾句离开前文就没有任何信息
_REFERENTIAL_TAIL = re.compile(r"以上|上述|前述|上文|前面")
# 回溯上限：正常切块 200~500 字，超过这个量级说明补过头了
_REFERENTIAL_EXPAND_MAX = 400

# FAQ 切块用「Q: 问句\nA: 正文」的格式，模型抄证据时会把这两个标记一起抄回来，
# 渲染出来就是「原文：A: 1 个月内的体检报告…」，用户看到的是一条坏掉的引用。
# 这两个标记属于切块格式，不是制度内容，展示前必须剥掉。
_EVIDENCE_Q_LINE = re.compile(r"^\s*Q\s*[：:][^\n]*\n\s*")
_EVIDENCE_A_MARK = re.compile(r"^\s*A\s*[：:]\s*")


def _clean_evidence(evidence: str) -> str:
    """去掉证据开头的 Q/A 行标记，只剥格式、不动内容。

    放在扩展之后调用：_expand_referential_evidence 要靠原文与证据的子串关系
    定位整句，先剥会改变匹配锚点。
    """
    text = _EVIDENCE_Q_LINE.sub("", evidence or "", count=1)
    return _EVIDENCE_A_MARK.sub("", text, count=1)


def _expand_referential_evidence(evidence: str, chunk_text: str) -> str:
    """把「以上…」这类收尾句补成自足的枚举段（仍是原文切片）。

    为什么必须做：PDF 切块常把多项内容用分号串成一段，末尾用
    「准备好以上材料，您就可以加入公司大家庭啦」收尾。模型倾向只抄这句收尾话
    —— 它确实是原文，逐字核验也能通过，但答案里没有一条具体信息，
    用户看到的是一句废话。这种情况下「作答」比拒答更糟。

    只做原文切片、不生成任何新内容，所以不破坏「逐字可回查」的前提。
    超出上限时保持原样：宁可答案短，也不能把整篇制度倒出来。
    """
    normalized = _normalize(evidence)
    if len(normalized) >= 40 or not _REFERENTIAL_TAIL.search(normalized):
        return evidence

    for clause in _CLAUSE_SPLIT.split(chunk_text):
        if normalized not in _normalize(clause):
            continue
        candidate = clause.strip()
        if len(candidate) > len(evidence) and len(candidate) <= _REFERENTIAL_EXPAND_MAX:
            return candidate
        return evidence

    return evidence


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]

# 规范类条款的特征词，用于「一条规范都没抽到」时决定要不要补抽
_RULE_SIGNAL = re.compile(r"需|应|必须|不得|严禁|禁止|要求|审批|报备|登记")


def _chunks_to_sweep(
    question: str,
    labeled_chunks: list[tuple[int, str]],
    records: list[dict],
) -> list[tuple[int, str]]:
    """找出「疑似被模型跳读」的片段，用于补抽一次。

    判据分两类，都要同时满足「句子与问题有词面交集」：
    - 处罚类：句子里有处罚用语，而这个片段还没产出过处罚记录；
    - 规范类：整批记录里一条规范都没有，且句子里有「需/应/必须/严禁」这类用语。

    为什么需要这一步：实测同一批次里，长片段末尾的条款会被漏掉
    ——「饲养宠物会有什么处罚」只抽到乐捐、漏掉另一份制度的扣分，
    而那句话在 511 字片段的最后一行；「我带宠物上班」则常漏掉
    「因公需求携带…需提前报备」这条例外款。把范围缩小到这些片段重抽，
    比放宽校验规则更安全：内容仍然要过逐字核验。上限 3 段、只跑一次，
    再多就退化成「又跑一遍全量抽取」，补捞也就失去意义了。
    """
    grams = _query_bigrams(question)
    if not grams:
        return []

    penalty_chunks = {
        source
        for record in records
        if record["kind"] == "penalty"
        for source in record["sources"]
    }
    rule_chunks = {
        source
        for record in records
        if record["kind"] == "rule"
        for source in record["sources"]
    }
    need_rules = not any(record["kind"] == "rule" for record in records)

    candidates: list[tuple[int, int, str]] = []
    for index, text in labeled_chunks:
        best_hit = 0
        for sentence in _sentences(text):
            normalized = _normalize(sentence)
            hit = sum(1 for gram in grams if gram in normalized)
            if not hit:
                continue
            if _PENALTY_SIGNAL.search(sentence) and index not in penalty_chunks:
                best_hit = max(best_hit, hit)
            elif need_rules and _RULE_SIGNAL.search(sentence) and index not in rule_chunks:
                best_hit = max(best_hit, hit)
        if best_hit:
            candidates.append((best_hit, index, text))

    # 命中问题关键词越多的句子越可能相关，先补它
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [(index, text) for _, index, text in candidates[:3]]


def _citations(sources: list[int]) -> str:
    return "".join(f"[文档{s}]" for s in sorted(sources))


def _common_phrase(actions: list[str]) -> str:
    """求多条处罚行为名共有的最长子串（≥3 字）。

    用途：两条罚则分别写「饲养宠物（鱼类除外）」和「饲养任何宠物，鱼类除外」时，
    用它说清「这些处罚针对的是同一个原文行为」，避免被读成两件不同的事。
    """
    if len(actions) < 2 or not actions[0]:
        return ""

    head = actions[0]
    for length in range(len(head), 2, -1):
        for start in range(0, len(head) - length + 1):
            candidate = head[start:start + length]
            if all(candidate in action for action in actions[1:]):
                if candidate.strip("，。；、,.;:"):
                    return candidate

    return ""


def _strip_tail(text: str) -> str:
    """去掉句末标点，避免「「…登记。」」这种套娃标点。"""
    return text.rstrip().rstrip("。；;，,")


def _display(text: str) -> str:
    """把原文里的换行去掉、连续空格压成一个。

    PDF 抽出的正文按版式换行，模型照抄时会保留「进入公司的\\n情况除外」这种断行；
    直接渲染会把一条要点拆成两行。只处理空白，不动字词，仍是逐字原文。
    """
    return _MULTI_SPACE.sub(" ", _LINE_BREAK.sub("", text or "")).strip()


def _trim_action(action: str, consequence: str) -> str:
    """行为名里混进了后果句时，把后果那段裁掉，只留行为。

    实测模型会把整句「若发现…，涉事责任人将乐捐200元/次…」当成行为名，
    渲染出来就是「行为『若发现…乐捐200元/次』：乐捐200元/次」——同一句罚则
    在答案里显示两遍。裁到后果句前最近的标点为止，留下的仍是原文片段，
    不引入任何新内容。
    """
    normalized_action = _normalize(action)
    normalized_consequence = _normalize(consequence)
    if not normalized_consequence or len(normalized_consequence) < 4:
        return action

    position = normalized_action.find(normalized_consequence)
    if position <= 0:
        return action

    head = normalized_action[:position]
    cut = max(head.rfind(char) for char in "，,；;。!")
    if cut >= 4:
        head = head[:cut]

    head = _strip_tail(head)
    if len(_normalize(head)) < 4:
        return action

    return head


def _render_answer(records: list[dict], gaps: list[str]) -> str:
    """由核验过的记录拼出答案。这里不调用模型，也不生成任何新事实。

    规范类记录直接引用原文证据整句：模型对「行为/条件/后果」的字段归属
    并不可靠（A3 实测固定槽位会把甲条款塞进乙槽位），而证据句是逐字核验过的，
    直接引它既不丢信息，也不依赖模型把内容放进哪个字段。
    处罚类记录必须写清「行为 → 后果」，因为这是用户要看的那层对应关系。
    """
    rules = [record for record in records if record["kind"] == "rule"]
    penalties = [record for record in records if record["kind"] == "penalty"]

    blocks: list[list[str]] = []

    if rules:
        lines = ["规定明确的部分："]
        for record in rules:
            lines.append(
                f"- 原文：「{_strip_tail(_display(record['evidence']))}」 "
                f"{_citations(record['sources'])}"
            )
        blocks.append(lines)

    if penalties:
        lines = ["相关处罚及适用范围："]
        for record in penalties:
            consequence = _strip_tail(_display(record["consequence"]))
            action = _strip_tail(_display(record["action"]))
            # 行为名里混进后果句时先裁掉，否则同一句罚则会显示两遍
            action = _strip_tail(_trim_action(action, consequence))
            if record["condition"]:
                action = f"{action}（{_display(record['condition'])}）"
            lines.append(
                f"- 原文规定的行为「{action}」："
                f"{consequence} {_citations(record['sources'])}"
            )
        # 只有各条行为名实质相同（同一条规定被抽成了多次）才说「均针对该行为」。
        # 若是并行的多条违规共用一句罚则，这句话会把不同行为说成一个，属误导。
        actions = [_normalize(record["action"]) for record in penalties]
        if len(penalties) >= 2 and all(
            _is_subsequence(other, actions[0]) or _is_subsequence(actions[0], other)
            for other in actions[1:]
        ):
            shared = _common_phrase([record["action"] for record in penalties])
            if shared:
                lines.append(
                    f"- 以上 {len(penalties)} 条处罚在原文中对应的行为都包含"
                    f"「{shared}」，均针对该行为，不涉及其他行为。"
                )
        blocks.append(lines)

    if gaps:
        lines = ["原文未明确的部分："]
        for gap in gaps:
            lines.append(f"- 现有资料未明确说明：{gap}")
        blocks.append(lines)

    rendered: list[str] = []
    for index, block in enumerate(blocks):
        if index:
            rendered.append("")
        rendered.append(f"{_CN_NUMERALS[index]}、{block[0]}")
        rendered.extend(block[1:])

    return "\n".join(rendered)



# ==========================================================
# 服务实现
# ==========================================================
class RAGService:
    """无状态服务：依赖由外部注入，可安全地被多个请求共享。"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or RAGConfig()

        # 并发上限夹紧到合法区间：若配成 0 或负数，Semaphore(0) 会让所有请求被拒
        raw_limit = self.config.max_concurrency
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            logger.warning(
                "max_concurrency=%r 非法，回退为 %s", raw_limit, _MIN_CONCURRENCY
            )
            limit = _MIN_CONCURRENCY

        if limit < _MIN_CONCURRENCY:
            logger.warning(
                "max_concurrency=%s 过小，回退为 %s", raw_limit, _MIN_CONCURRENCY
            )
            limit = _MIN_CONCURRENCY
        elif limit > _MAX_CONCURRENCY:
            logger.warning(
                "max_concurrency=%s 过大，夹紧为 %s", raw_limit, _MAX_CONCURRENCY
            )
            limit = _MAX_CONCURRENCY

        self._effective_concurrency = limit

        # 相似度阈值夹紧到 [0, 1]：
        # 写成负数等于没过滤，写成 2 会把所有片段都丢掉、让服务永远只回「无法回答」。
        # 这类配置错误不该以「答不出来」的形式表现，所以这里夹紧并留一条日志。
        raw_score = self.config.min_score
        try:
            min_score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning(
                "min_score=%r 非法，回退为 0.0（不过滤）", raw_score
            )
            min_score = 0.0

        if not math.isfinite(min_score):
            logger.warning(
                "min_score=%r 不是有限数，回退为 0.0（不过滤）", raw_score
            )
            min_score = 0.0
        elif min_score < 0.0:
            logger.warning("min_score=%s 小于 0，按 0 处理（不过滤）", min_score)
            min_score = 0.0
        elif min_score > 1.0:
            logger.warning("min_score=%s 大于 1，夹紧为 1.0", min_score)
            min_score = 1.0

        self._effective_min_score = min_score

        # 惰性创建：确保信号量绑定到真正运行时的事件循环
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._effective_concurrency)
        return self._semaphore

    async def answer(
        self,
        query: str,
        history: list[dict],
        top_k: int,
    ) -> RAGResult:
        """
        执行一次完整的 RAG 问答。

        Args:
            query: 用户当前问题
            history: 该会话的历史消息（已按时间正序）
            top_k: 检索候选数量

        Returns:
            RAGResult(answer, sources)

        Raises:
            RetrievalUnavailable: 检索阶段失败
            GenerationFailed: 生成阶段失败
        """
        # 限流：拿不到名额就快速失败，让调用方知道"忙"，而不是排队几十秒
        semaphore = self._get_semaphore()
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=_ACQUIRE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise ServiceBusy("当前并发已满") from exc

        try:
            # 一、构建增强查询
            enhanced_query = self._build_contextual_query(query, history)

            # 二、向量检索
            t_retrieval = time.perf_counter()
            try:
                raw_results = await self.vector_store.search(
                    query=enhanced_query,
                    top_k=top_k,
                )
            except Exception as exc:
                logger.exception("知识检索失败")
                raise RetrievalUnavailable("知识检索失败") from exc
            retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

            # 三、低分过滤
            chunks = self._filter_by_score(raw_results)

            # 「检索到了、但全部不达标」→ 不调 LLM，直接固定拒答。
            # 注意条件是 raw_results 非空：知识库本身就是空的时候（raw_results 为空）
            # 仍然走原来的生成路径，避免 min_score=0 的默认配置改变既有行为。
            if raw_results and not chunks:
                logger.info(
                    "检索片段全部低于阈值 %.3f，跳过生成直接拒答", self._effective_min_score
                )
                return RAGResult(
                    answer=_NO_CONTEXT_ANSWER,
                    sources=[],
                    retrieval_ms=round(retrieval_ms, 1),
                    generation_ms=0.0,
                )

            # 四、两阶段回答：抽取 → 校验 → 渲染
            t_generation = time.perf_counter()
            try:
                answer_text, used_sources = await self._answer_two_stage(
                    question=enhanced_query,
                    chunks=chunks,
                )

                # 按需改写：只在「拒答」时触发一次。
                # 用户用语与库内问法差太远时（「面试过了要干啥」vs
                # 「面试通过啦，我需要准备什么材料来办岗前培训呀」），抽取器会
                # 判定无关而空手。此时把问题对齐到库内问法，并**用新问法重新检索**
                # —— 检索和抽取必须用同一句话：只改写抽取不重检索，上下文还是旧问法
                # 召回的邻居块，实测仍然空手（0.6391 vs 0.7166 的差异就在邻居块上）。
                if not used_sources:
                    retry_answer, retry_sources, retry_ms = (
                        await self._retry_with_aligned_question(query, chunks, top_k)
                    )
                    if retry_sources:
                        answer_text, used_sources = retry_answer, retry_sources
                        retrieval_ms += retry_ms
            except Exception as exc:
                logger.exception("LLM 生成失败")
                raise GenerationFailed("AI 回答生成失败") from exc
            generation_ms = (time.perf_counter() - t_generation) * 1000

            return RAGResult(
                answer=answer_text,
                # sources 用核验过的那批片段：对外声称「依据这些片段回答」，
                # 就必须是真正支撑答案的那几条；拒答时为空，
                # 避免出现「答不出却列了一堆来源」的观感。
                sources=used_sources,
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(generation_ms, 1),
            )
        finally:
            # 无论成功、失败还是异常，都必须归还名额，否则并发会永久耗尽
            semaphore.release()

    # ----------------------------------------
    # 内部步骤
    # ----------------------------------------
    def _filter_by_score(self, results: list[dict]) -> list[dict]:
        """丢弃相似度低于阈值的检索结果。

        约定：
        - 阈值为 0（默认）时**原样返回**，不复制不过滤 —— 保证默认配置下
          行为与改造前完全一致；
        - score 缺失或不是数字时**保留**：宁可多给模型一段无关内容，
          也不要在上游数据结构变化时把正确答案误删。

        score 语义见 VectorStore.search()：1 - 余弦距离，越大越相似。
        """
        threshold = self._effective_min_score
        if threshold <= 0.0 or not results:
            return results

        kept: list[dict] = []
        dropped_scores: list[float] = []

        for item in results:
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                kept.append(item)
                continue

            if score >= threshold:
                kept.append(item)
            else:
                dropped_scores.append(score)

        if dropped_scores:
            logger.info(
                "低分过滤：保留 %s/%s 段，阈值 %.3f，被丢弃片段最高分 %.3f",
                len(kept),
                len(results),
                threshold,
                max(dropped_scores),
            )

        return kept

    @staticmethod
    def _build_contextual_query(
        current_message: str,
        history: list[dict],
    ) -> str:
        """
        基于历史消息构建增强查询（轻量指代消解）。

        规则：
        1. 无历史、或当前问题较长（自带完整语义）→ 原样返回，不改写。
        2. 短句且含指代词/承接词（如「那第二次呢？」）→ 拼接上一条用户问题，
           让向量检索能落到正确的主题上。

        实测依据（同一知识库、top_k=5）：
        - 「那第二次呢？」单独检索 → top1=0.6016，命中无关的章节标题；
        - 拼接上文「我上班打游戏扣多少钱」后 → top1=0.6881，正确命中
          「严禁员工上班期间打游戏…乐捐200」。
        """
        text = (current_message or "").strip()
        if not text or not history:
            return current_message

        is_short = len(text) <= _FOLLOWUP_MAX_LEN
        has_marker = any(marker in text for marker in _FOLLOWUP_MARKERS)
        if not (is_short and has_marker):
            return current_message

        # history 取的是"当前问题保存之前"的消息，最近一条 user 消息即上文
        last_user = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user = str(msg.get("content") or "").strip()
                break

        if not last_user:
            return current_message

        return f"{last_user} {text}"

    async def _request_extraction(
        self,
        question: str,
        labeled_chunks: list[tuple[int, str]],
    ) -> str:
        """发起一次抽取调用。labeled_chunks 的编号直接用作答案里的 [文档N]。"""
        context_text = "\n\n".join(
            f"[文档{index}]\n{text}" for index, text in labeled_chunks
        )

        messages = [
            {"role": "system", "content": f"{_EXTRACTION_SYSTEM_PROMPT}\n{context_text}"},
            {"role": "user", "content": f"用户问题：{question}"},
        ]

        # 同步客户端交给线程池，避免阻塞事件循环；format 让 Ollama 用
        # JSON Schema 构建解码语法，输出必须是这个结构。
        # temperature 固定 0：抽取是「照抄原文」的确定性任务，默认 0.2 的
        # 采样噪音会让同一片段两次抽出不同记录（一次带清单、一次只带收尾句）。
        return await asyncio.to_thread(
            self.llm_client.chat,
            messages=messages,
            format=_EXTRACTION_SCHEMA,
            num_predict=_EXTRACTION_NUM_PREDICT,
            temperature=0.0,
        )

    async def _retry_with_aligned_question(
        self,
        question: str,
        chunks: list[dict],
        top_k: int,
    ) -> tuple[str, list[dict], float]:
        """拒答后的唯一一次补救：对齐问法 → 重新检索 → 重新抽取。

        为什么必须重新检索：改写句的 Top-1 分数更高（实测 0.6391 → 0.7166），
        带出来的邻居块也不同。只改写抽取、不重新检索，上下文还是旧问法召回的那批
        片段，抽取器照样空手。

        Returns:
            (答案, 支撑片段, 本次检索片段, 本次检索耗时 ms)。
            任何一步没拿到东西都返回空结果，由调用方沿用原来的拒答。
        """
        if not self.config.enable_align_rewrite or self.vector_store is None:
            return _NO_CONTEXT_ANSWER, [], 0.0

        aligned = await self._align_question(question, chunks)
        if not aligned:
            return _NO_CONTEXT_ANSWER, [], 0.0

        logger.info("按需改写问题并重新检索：%r -> %r", question[:40], aligned[:40])

        t_retrieval = time.perf_counter()
        try:
            raw_results = await self.vector_store.search(query=aligned, top_k=top_k)
        except Exception:
            # 补救路径失败不能把原本的「拒答」变成「报错」
            logger.warning("改写后重新检索失败，沿用原拒答", exc_info=True)
            return _NO_CONTEXT_ANSWER, [], 0.0
        retry_ms = (time.perf_counter() - t_retrieval) * 1000

        retry_chunks = self._filter_by_score(raw_results)
        if not retry_chunks:
            return _NO_CONTEXT_ANSWER, [], retry_ms

        try:
            answer_text, sources = await self._answer_two_stage(aligned, retry_chunks)
        except Exception:
            logger.warning("改写后重新抽取失败，沿用原拒答", exc_info=True)
            return _NO_CONTEXT_ANSWER, [], retry_ms

        if not sources:
            return _NO_CONTEXT_ANSWER, [], retry_ms

        return answer_text, sources, retry_ms

    def _parse_extraction_or_none(self, raw: str, stage: str) -> Optional[dict]:
        try:
            return _parse_extraction(raw)
        except ValueError as exc:
            logger.error(
                "抽取结果不可用（%s），降级为拒答：%s 原始输出前 120 字=%r",
                stage,
                exc,
                (raw or "")[:120],
            )
            return None

    @staticmethod
    def _alignment_references(chunks: list[dict]) -> list[str]:
        """取前几条片段的库内问法，作为改写的候选锚点。

        FAQ 类切块有 question 字段，直接用它；其它文档没有，就取正文开头，
        当作「库内是这么说的」的措辞参考。

        给多条而不是只给 Top-1：检索把哪个块排第一并不可靠
        （「面试完了该干嘛」的离职块 0.6270 反而压过正确的材料块 0.6108），
        把候选交给模型按字面挑，比钉死一条稳。
        """
        references: list[str] = []
        for chunk in chunks[:_ALIGN_REFERENCE_COUNT]:
            metadata = chunk.get("metadata") or {}
            reference = (metadata.get("question") or "").strip()
            if not reference:
                head = _LINE_BREAK.sub(" ", chunk.get("text") or "").strip()
                reference = head[:_ALIGN_REFERENCE_CHARS]
            if reference and reference not in references:
                references.append(reference)
        return references

    async def _align_question(
        self,
        question: str,
        chunks: list[dict],
    ) -> Optional[str]:
        """按需改写：把用户的口语问题对齐到库内已有问法。

        只在整轮抽取判定「拒答」之后调用一次（见 _retry_with_aligned_question）。
        正常问题不走这里，所以正常路径没有额外延迟；改写只发生在「本来就要拒答」
        的分支上，即使改偏，最差结果也只是继续拒答，不会比不改更糟。

        Returns:
            可用的改写问句；判定不是同一件事、输出不可用、或改写跑题时返回 None。
        """
        if not self.config.enable_align_rewrite:
            return None

        references = self._alignment_references(chunks)
        if not references:
            return None

        reference_text = "\n".join(
            f"{index}. {text}" for index, text in enumerate(references, start=1)
        )

        try:
            raw = await asyncio.to_thread(
                self.llm_client.chat,
                messages=[
                    {"role": "system", "content": _ALIGN_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{question}\n"
                            f"候选参考问法（择一）：\n{reference_text}"
                        ),
                    },
                ],
                format=_ALIGN_SCHEMA,
                num_predict=_ALIGN_NUM_PREDICT,
                # 判定「是不是同一件事」不能有随机性：0.2 的温度会让同一句
                # 话这次判 true、下次判 false，表现为时答时不答。
                temperature=0.0,
            )
        except Exception:
            # 改写是兜底路径，失败就当没有，绝不能把可回答的问题变成异常
            logger.warning("问句对齐调用失败，跳过改写", exc_info=True)
            return None

        try:
            payload = json.loads(_JSON_FENCE.sub("", (raw or "")).strip())
        except (ValueError, TypeError):
            logger.info("问句对齐输出不是合法 JSON，跳过改写：%r", (raw or "")[:120])
            return None

        if not isinstance(payload, dict):
            logger.info("问句对齐输出不是对象，跳过改写：%r", (raw or "")[:120])
            return None

        # schema 里没有 same_topic 字段，模型只输出 rewritten。
        # 「不需要改写」的约定是原样返回用户问题，下面的 candidate == question
        # 会把它当成没有改写直接跳过，所以正常路径不受影响。
        candidate = " ".join(str(payload.get("rewritten") or "").split())
        candidate = candidate.strip(" \"'“”‘’《》「」")

        if not candidate:
            return None
        if len(candidate) > _ALIGN_MAX_CHARS:
            logger.info("改写问句过长（%s 字），跳过改写：%r", len(candidate), candidate[:40])
            return None
        if candidate == question:
            return None

        # 防跑题：改写后必须与用户原问题存在共同的二字片段。
        # 「面试过了之后要干啥」→「面试过了之后需要准备什么」保留了「面试」；
        # 若改写把对象整个换掉，这里直接拦下，不让它去抽取别的内容。
        if not (_query_bigrams(candidate) & _query_bigrams(question)):
            logger.info(
                "改写问句与用户问题无共同片段，判为跑题并跳过：%r -> %r",
                question[:40],
                candidate[:40],
            )
            return None

        return candidate

    async def _answer_two_stage(
        self,
        question: str,
        chunks: list[dict],
    ) -> tuple[str, list[dict]]:
        """两阶段回答：模型只负责抽取，答案由代码拼装。

        Returns:
            (答案文本, 真正支撑答案的片段列表)。拒答时片段列表为空。

        降级策略（不抛异常，直接给安全拒答）：
        - 抽取结果不是合法 JSON；
        - 抽取记录全部核验失败（证据找不到 / 编号对不上 / 处罚无据）；
        - 记录与问题连一个二字片段都不重合。
        这几种情况都不会退回自由生成 —— 那正是本次要关掉的路径。
        """
        used_chunks = chunks[: self.config.max_sources]
        if not used_chunks:
            return _NO_CONTEXT_ANSWER, []

        labeled = [(index + 1, chunk["text"]) for index, chunk in enumerate(used_chunks)]
        chunk_texts = dict(labeled)

        raw = await self._request_extraction(question, labeled)
        payload = self._parse_extraction_or_none(raw, "首次抽取")
        if payload is None:
            return _NO_CONTEXT_ANSWER, []

        records = _validate_records(payload.get("records"), chunk_texts)
        gaps = _validate_gaps(payload.get("gaps"), question, records)

        # 补抽：只针对「句子既沾问题关键词、又像处罚（或整批一条规范都没有）」、
        # 却还没产出对应记录的片段。最多再跑一次，结果同样要过逐字核验。
        sweep = _chunks_to_sweep(question, labeled, records)
        if sweep:
            logger.info(
                "抽取补捞未覆盖片段：%s（已覆盖 %s）",
                [index for index, _ in sweep],
                sorted({s for record in records for s in record["sources"]}),
            )
            retry_raw = await self._request_extraction(question, sweep)
            retry_payload = self._parse_extraction_or_none(retry_raw, "补抽")
            if retry_payload is not None:
                extra = _validate_records(retry_payload.get("records"), dict(sweep))
                if extra:
                    records = _merge_duplicate_records(records + extra)
                if not gaps:
                    gaps = _validate_gaps(retry_payload.get("gaps"), question, records)

        if not records:
            logger.info("抽取没有得到可核验的记录，降级为拒答：question=%r", question[:40])
            return _NO_CONTEXT_ANSWER, []

        # 只记结构化的关键字段，便于线上排查「为什么这条没被用上」
        logger.debug(
            "核验后的记录：%s",
            json.dumps(
                [
                    {
                        "kind": record["kind"],
                        "action": record["action"],
                        "condition": record["condition"],
                        "consequence": record["consequence"],
                        "evidence": record["evidence"],
                        "sources": record["sources"],
                    }
                    for record in records
                ],
                ensure_ascii=False,
            ),
        )

        if not _topically_related(question, records, chunk_texts):
            logger.info(
                "抽取记录与问题无词面交集，降级为拒答：question=%r 记录数=%s",
                question[:40],
                len(records),
            )
            return _NO_CONTEXT_ANSWER, []

        # 「原文未明确」这一块由代码补齐：模型填的 gaps 有方差，
        # 而「原文只规定因公、非因公没写」是能从已核验记录里直接推出来的。
        # gap 可能来自首次抽取，也可能来自补抽：首次抽取时若记录为空，
        # _validate_gaps 拿不到可比的记录，过滤会失效；补抽分支又只在
        # gaps 为空时才重筛。所以这里以最终 records 为准再筛一遍。
        gaps = [
            gap
            for gap in gaps
            if not _gap_contradicts_records(gap, records)
            and not _gap_off_topic(question, gap)
        ]
        gaps = _merge_gaps(gaps, _derived_gaps(question, records))

        answer_text = _render_answer(records, gaps)

        # 引用是代码生成的，这里的检查只是兜底观测（正常不会告警）。
        self._check_citations(answer_text, len(used_chunks))

        return answer_text, used_chunks

    @staticmethod
    def _check_citations(answer: str, source_count: int) -> None:
        """检查回答里的来源标注，只写日志、不改写内容。

        不改写的理由：这段文本会直接落库并返回给用户，在服务层用正则加工
        很容易把正常内容切坏（用户问题本身可能就带方括号）。
        这里只负责把「没标来源」和「标了不存在的编号」变成可观测的信号，
        要不要重试或降级由后续策略决定。
        """
        if source_count <= 0:
            return

        cited = {int(match) for match in _CITATION_PATTERN.findall(answer)}

        if not cited:
            logger.warning(
                "回答未标注来源：sources=%s 回答前40字=%r",
                source_count,
                answer[:40],
            )
            return

        invalid = sorted(n for n in cited if n < 1 or n > source_count)

        if invalid:
            logger.warning(
                "回答引用了不存在的片段编号 %s（本次只有 %s 段）",
                invalid,
                source_count,
            )
