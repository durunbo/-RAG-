"""流程阶段检索：在 Dense 找到流程文档后，按阶段规则筛选 Chunk。"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 支持阿拉伯数字和常见中文数词，例如 14、二十一、三十、一百、半。
_NUMBER = r"(?:\d+(?:\.\d+)?|半|[零〇一二两三四五六七八九十百千万]+)"
_UNIT = r"(?:个月|天|日|周|月)"

# 用户问题只识别有明确阶段语境的表达，避免把“提前3天”误认为阶段。
_QUERY_PATTERNS = [
    re.compile(rf"(?P<raw>入职\s*(?:第\s*)?(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT}))"),
    re.compile(rf"(?P<raw>第\s*(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT}))"),
    re.compile(rf"(?P<raw>满\s*(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT}))"),
    re.compile(rf"(?P<raw>(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT})\s*(?:阶段|节点|时点))"),
    # 支持“半个月需要做什么”这种没有“入职”前缀的明确阶段问题。
    re.compile(r"(?P<raw>(?P<number>半)\s*个月(?=\s*(?:阶段|节点|时点|时|需要|要做|要完成|有哪些|做什么|任务|工作)))"),
]

# stage 元数据本身就是阶段标签，因此允许“7天”“1个月”等独立写法。
_CHUNK_PATTERNS = _QUERY_PATTERNS + [
    re.compile(rf"(?P<raw>(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT}))"),
]

# 只排除紧挨当前时间表达的上下文，不对整句话一票否决。
_NON_STAGE_CONTEXT = re.compile(r"(?:提前|每隔|每|间隔|延后|推迟|截止|距离|还有|不到|超过)\s*$")
_DAYS_PER_UNIT = {"天": 1.0, "日": 1.0, "周": 7.0, "月": 30.0, "个月": 30.0}


class ProcessStageRetrievalError(RuntimeError):
    """阶段检索已经触发，但读取数据或处理结果发生错误。"""


def _parse_number(text: str) -> Optional[float]:
    """将阿拉伯数字或常见中文整数转成数值。"""
    text = text.strip()
    if not text:
        return None
    if text == "半":
        return 0.5
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
              "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    total = section = current = 0

    # 示例：十四=10+4，二十一=2*10+1，一百二十三=100+20+3。
    for char in text:
        if char in digits:
            current = digits[char]
        elif char in units:
            section += (current or 1) * units[char]
            current = 0
        elif char == "万":
            section += current
            total += section * 10000
            section = current = 0
        else:
            return None
    return float(total + section + current)


def _normalise(text: Any) -> str:
    """只用于标题相等性判断，忽略空白和常见标点。"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s\u3000:：/\\|>＞、，,。．.!！?？;；（）()【】\[\]{}_-]+", "", text).casefold()


def _normalise_unit(unit: str) -> Optional[str]:
    if unit in ("天", "日"):
        return "天"
    if unit in ("月", "个月"):
        return "月"
    if unit == "周":
        return "周"
    return None


def _parse_stage_text(
    text: str,
    patterns: List[re.Pattern],
    reject_non_stage_context: bool,
) -> Optional[Dict[str, Any]]:
    """从一段文本中提取唯一阶段；多个不同阶段表达视为歧义，不猜测。"""
    if not isinstance(text, str) or not text.strip():
        return None

    candidates = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            number_text = groups.get("number")
            if not number_text:
                continue

            if reject_non_stage_context:
                before = text[max(0, match.start() - 8):match.start()]
                if _NON_STAGE_CONTEXT.search(before):
                    continue

            value = _parse_number(number_text)
            if value is None or value <= 0:
                continue

            # “半个月需要做什么”匹配时，正则中没有单独的 unit 捕获组。
            unit = _normalise_unit(groups.get("unit") or "月")
            if unit is None:
                continue

            candidates.append({
                "start": match.start(),
                "end": match.end(),
                "stage": {
                    "raw": (groups.get("raw") or match.group(0)).strip(),
                    "value": value,
                    "unit": unit,
                    "compare_days": value * _DAYS_PER_UNIT[unit],
                },
            })

    if not candidates:
        return None

    # 同一个表达可能被多条正则重复识别；优先保留起点靠前且更完整的匹配。
    candidates.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    unique = []
    last_end = -1
    for item in candidates:
        if item["start"] < last_end:
            continue
        unique.append(item)
        last_end = item["end"]

    # 多个互不重叠的时间通常表示对比问题，不擅自挑其中一个。
    if len(unique) != 1:
        return None
    return unique[0]["stage"]


class ProcessStageRetriever:
    """流程阶段检索器。只负责阶段筛选，不改动 Dense 检索和向量库结构。"""

    SOURCE_LOOKAHEAD = 3  # 检查 Dense 排名前3条，避免跨文档拼接阶段
    FLOAT_TOLERANCE = 1e-8

    def __init__(self, vector_store: Any):
        """注入现有 VectorStore；本类不创建 Embedding 模型或 Chroma 客户端。"""
        self.vector_store = vector_store

    def _extract_query_stage(self, query: str) -> Optional[Dict[str, Any]]:
        """解析用户问题；无法明确识别时返回 None，沿用原 Dense 结果。"""
        return _parse_stage_text(query, _QUERY_PATTERNS, reject_non_stage_context=True)

    def _extract_chunk_stage(self, chunk_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """只解析 metadata.stage，不从正文数字推断阶段。"""
        if not isinstance(chunk_item, dict):
            return None
        metadata = chunk_item.get("metadata")
        if not isinstance(metadata, dict):
            return None

        stage = metadata.get("stage", "")
        info = _parse_stage_text(stage, _CHUNK_PATTERNS, reject_non_stage_context=True)
        if info is None:
            return None
        return info

    def _find_process_source(self, dense_results: List[Dict[str, Any]]) -> Optional[str]:
        """定位唯一流程文档；前3条命中多个流程文件时保守地不做阶段筛选。"""
        process_files = []
        for result in dense_results[:self.SOURCE_LOOKAHEAD]:
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("document_type") != "process":
                continue

            source_file = metadata.get("source_file")
            if isinstance(source_file, str) and source_file.strip():
                source_file = source_file.strip()
                if source_file not in process_files:
                    process_files.append(source_file)

        return process_files[0] if len(process_files) == 1 else None

    @staticmethod
    def _split_heading_path(heading_path: Any) -> List[str]:
        """兼容 Chroma 中的字符串路径，以及测试中直接传入的列表。"""
        if isinstance(heading_path, list):
            return [p.strip() for p in heading_path if isinstance(p, str) and p.strip()]
        if isinstance(heading_path, str) and heading_path.strip():
            # VectorStore.add_chunks() 当前将 heading_path 用“ / ”拼接。
            return [p.strip() for p in re.split(r"\s*/\s*", heading_path) if p.strip()]
        return []

    def _extract_body_for_title_check(self, text: str, heading_path: Any) -> str:
        """尽可能提取 Embedding 文本中的正文，仅用于过滤纯标题块。"""
        # 兼容常见模板中的“正文：”“内容：”“【正文】”等正文标记。
        marker = re.search(
            r"(?:^|[\r\n|｜]\s*|\s+)"
            r"(?:(?:【|\[)(?:正文|内容|文本)(?:】|\])|(?:正文|内容|文本)\s*[:：])\s*",
            text,
        )
        if marker:
            return text[marker.end():].strip()

        # 若整个标题路径是文本前缀，只在前缀后存在分隔符时才剥离。
        if isinstance(heading_path, str) and heading_path.strip():
            prefix = heading_path.strip()
            if text.startswith(prefix):
                remainder = text[len(prefix):]
                if not remainder:
                    return ""
                if remainder[0].isspace() or remainder[0] in ":：|｜-/＞>、，。;；":
                    return remainder.lstrip(" \t\r\n:：|｜-/＞>、，。;；")
        # 无法确认正文边界时不做猜测，避免误删短任务。
        return text.strip()

    def _is_valid_task_chunk(self, chunk_item: Dict[str, Any]) -> bool:
        """
        判断 Chunk 是否包含有效任务内容。

        规则：
        1. 排除空文本。
        2. 排除纯阶段标题。
        3. 排除格式明确的章节标题和独立责任部门标签。
        4. 不再通过 heading_path 判断正文是不是标题，
           因为当前项目的 heading_path 可能包含正文自身。
        5. 不使用字符长度过滤，避免误删短任务。
        """
        if not isinstance(chunk_item, dict):
            return False

        text = chunk_item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return False

        metadata = chunk_item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        stage = metadata.get("stage", "")
        if not isinstance(stage, str):
            stage = ""

        heading_path = metadata.get("heading_path", "")

        # 尽量从 Embedding 文本中提取正文。
        body = self._extract_body_for_title_check(
            text,
            heading_path,
        ).strip()

        if not body:
            return False

        # 统一空格、标点，避免“入职30天”和“入职 30 天”
        # 因为格式不同而无法比较。
        body_norm = _normalise(body)
        text_norm = _normalise(text)
        stage_norm = _normalise(stage)

        # 规则一：纯阶段标题不作为任务答案。
        if stage_norm and body_norm == stage_norm:
            return False

        # 兼容某些情况下，存储文本本身就是阶段标题。
        if stage_norm and text_norm == stage_norm:
            return False

        # 规则二：过滤具有明确格式的独立标题。
        # 这些规则只针对结构明确的标题，不依靠正文长短猜测。
        structural_title_patterns = [
            # 例如：1.1 入职手续与资料归档、2.1 岗位技能达标评估
            r"\d+(?:\.\d+)+\s*[^。！？]{1,60}",

            # 例如：一、适用范围与总则
            r"[一二三四五六七八九十]+、[^。！？]{1,60}",

            # 例如：第六章、第二部分
            r"第[一二三四五六七八九十百千\d]+[章节篇部分]\s*.*",

            # 单独的责任部门说明不是独立任务。
            r"责任部门\s*[:：]\s*.+",
        ]

        for pattern in structural_title_patterns:
            if re.fullmatch(pattern, body):
                return False

        # 关键修正：
        # 不再判断 body_norm 是否出现在 heading_path 中。
        # 当前项目中，真实任务正文也可能出现在 heading_path 里。
        return True

    @staticmethod
    def _chunk_sort_key(chunk: Dict[str, Any]) -> Tuple[int, int]:
        """chunk_index 有效时升序排列；缺失或异常的索引放最后。"""
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            return (1, 0)
        index = metadata.get("chunk_index")
        if isinstance(index, bool):
            return (1, 0)
        if isinstance(index, int) and index >= 0:
            return (0, index)
        if isinstance(index, str) and index.isdigit():
            return (0, int(index))
        return (1, 0)

    def _select_target_stages(
        self,
        all_chunks: List[Dict[str, Any]],
        target_stage: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """精确阶段优先；否则选前后最近阶段；越界只取单侧边界。"""
        if not all_chunks or not isinstance(target_stage, dict):
            return []

        target_days = target_stage.get("compare_days")
        target_unit = target_stage.get("unit")
        target_value = target_stage.get("value")
        if not isinstance(target_days, (int, float)) or not isinstance(target_value, (int, float)):
            return []
        if target_unit not in {"天", "周", "月"}:
            return []

        # 先过滤纯标题，再建立阶段组，避免没有任务的标题成为相邻阶段候选。
        groups: Dict[Tuple[str, float], Dict[str, Any]] = {}
        for chunk in all_chunks:
            stage = self._extract_chunk_stage(chunk)
            if stage is None or not self._is_valid_task_chunk(chunk):
                continue

            # 阶段身份使用“单位+数值”；比较顺序才使用换算后的天数。
            key = (stage["unit"], round(float(stage["value"]), 8))
            if key not in groups:
                groups[key] = {"stage": stage, "chunks": []}
            groups[key]["chunks"].append(chunk)

        if not groups:
            return []

        ordered_groups = sorted(
            groups.values(),
            key=lambda g: (g["stage"]["compare_days"], g["stage"]["unit"], g["stage"]["value"]),
        )
        target_key = (target_unit, round(float(target_value), 8))

        # 1. 单位和值都一致：只返回该阶段。
        if target_key in groups:
            selected_groups = [groups[target_key]]
        else:
            # 2. 单位不同但时长相同，例如“30天”对应文档中唯一的“1个月”。
            same_time = [
                g for g in ordered_groups
                if abs(g["stage"]["compare_days"] - target_days) <= self.FLOAT_TOLERANCE
            ]
            if len(same_time) == 1:
                selected_groups = same_time
            elif len(same_time) > 1:
                # 同时存在多个等价阶段标签时不猜测，避免错误合并。
                logger.warning("目标时间 %.2f 天对应多个阶段标签，返回空结果。", target_days)
                return []
            else:
                # 3. 找严格小于目标的最近阶段、严格大于目标的最近阶段。
                earlier = [g["stage"]["compare_days"] for g in ordered_groups
                           if g["stage"]["compare_days"] < target_days]
                later = [g["stage"]["compare_days"] for g in ordered_groups
                         if g["stage"]["compare_days"] > target_days]
                selected_times = set()
                if earlier:
                    selected_times.add(max(earlier))
                if later:
                    selected_times.add(min(later))
                # 若目标时间在所有阶段之前/之后，结果自然只含最早/最晚阶段。
                selected_groups = [
                    g for g in ordered_groups
                    if any(abs(g["stage"]["compare_days"] - d) <= self.FLOAT_TOLERANCE
                           for d in selected_times)
                ]

        # 返回所选阶段的全部任务 Chunk，并按原文 chunk_index 顺序整理、去重。
        result, seen = [], set()
        for group in selected_groups:
            for chunk in group["chunks"]:
                metadata = chunk.get("metadata", {})
                chunk_id = chunk.get("id")
                if chunk_id:
                    key = ("id", chunk_id)
                else:
                    key = ("fallback", metadata.get("source_file"),
                           metadata.get("chunk_index"), chunk.get("text", ""))
                if key not in seen:
                    seen.add(key)
                    result.append(chunk)
        result.sort(key=self._chunk_sort_key)
        return result

    def retrieve(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """对外入口。

        None = 未触发阶段检索，上层使用 Dense 结果；
        列表 = 已执行阶段筛选，空列表表示没有可用阶段内容；
        已触发后发生错误则抛出 ProcessStageRetrievalError，不静默回退。
        """
        if not isinstance(query, str) or not query.strip():
            return None
        if not isinstance(dense_results, list) or not dense_results:
            return None

        # 第一步：问题未指定明确阶段时，不干预 Dense。
        target_stage = self._extract_query_stage(query)
        if target_stage is None:
            return None

        # 第二步：只有定位到唯一流程文件时才继续。
        source_file = self._find_process_source(dense_results)
        if source_file is None:
            return None

        # 第三步：读取该文件的全部流程 Chunk，不局限于 Dense 返回的少数几条。
        try:
            all_chunks = self.vector_store.get_process_chunks(source_file)
        except Exception as exc:
            raise ProcessStageRetrievalError(
                f"读取流程文档 {source_file!r} 的 Chunk 失败"
            ) from exc

        if not all_chunks:
            logger.warning("流程文档 %r 中没有可读取的流程 Chunk。", source_file)
            return []

        # 第四步：选阶段。阶段检索已触发，处理失败必须暴露给调用方。
        try:
            return self._select_target_stages(all_chunks, target_stage)
        except Exception as exc:
            raise ProcessStageRetrievalError(
                f"筛选流程文档 {source_file!r} 的阶段 Chunk 失败"
            ) from exc