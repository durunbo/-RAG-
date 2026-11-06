#文档类型自动识别
import os
import re
from typing import List, Dict
from dataclasses import dataclass, field
from src.models import DocumentBlock


# ==========================================================
# 一、分类结果结构：可追溯、可调试、支持混合类型
# ==========================================================

@dataclass
class ClassificationResult:
    """文档分类结果，包含完整判断依据"""
    primary_type: str                    # 主类型，用于选择切块策略
    secondary_types: List[str] = field(default_factory=list)  # 次要类型标签
    scores: Dict[str, int] = field(default_factory=dict)     # 各类型得分明细
    matched_evidence: List[str] = field(default_factory=list) # 命中的证据列表
    needs_review: bool = False            # 是否需要人工复核
    is_ignore: bool = False               # 是否为临时/无效文件，直接跳过


# ==========================================================
# 二、评分规则配置：强特征高分、弱特征低分，避免误判
# ==========================================================

# 每类文档的评分规则：(匹配规则, 得分, 证据描述)
# 强特征（+3分）：结构级特征，几乎可以确定类型
# 中特征（+2分）：典型内容特征
# 弱特征（+1分）：关键词特征，仅作辅助

_SCORING_RULES = {
    "faq": [
        # 强特征：问答结构标记
        (lambda text: bool(re.search(r'^Q[：:]', text, re.M)), 3, "存在 Q: 问答标记"),
        (lambda text: bool(re.search(r'^A[：:]', text, re.M)), 3, "存在 A: 回答标记"),
        # 中特征：高频问句
        (lambda text: text.count("？") > 5, 2, "问号数量较多"),
        # 弱特征：文件名关键词
        (lambda name: any(kw in name for kw in ["faq", "问答", "常见问题", "f&q"]), 1, "文件名包含FAQ类关键词"),
    ],

    "policy": [
        # 强特征：制度章节结构
        (lambda text: bool(re.search(r'第[一二三四五六七八九十百千\d]+章', text)), 3, "存在第X章结构"),
        (lambda text: bool(re.search(r'第[一二三四五六七八九十百千\d]+条', text)), 3, "存在第X条结构"),
        # 中特征：制度类高频词
        (lambda text: any(kw in text for kw in ["应当", "不得", "必须", "禁止", "处罚", "规定", "责令"]), 2, "包含制度规范类用词"),
        # 弱特征：文件名关键词
        (lambda name: any(kw in name for kw in ["制度", "规定", "管理办法", "准则", "条例", "守则"]), 1, "文件名包含制度类关键词"),
    ],

    "process": [
        # 强特征：流程步骤标记
        (lambda text: bool(re.search(r'第[一二三四五六七八九十百千\d]+步', text)), 3, "存在第X步流程标记"),
        (lambda text: bool(re.search(r'入职\s*\d+天', text)), 3, "存在入职X天时间节点"),
        # 中特征：流程类高频词
        (lambda text: any(kw in text for kw in ["审批", "提交", "审核", "时间节点", "责任人", "完成", "启动"]), 2, "包含流程类用词"),
        # 弱特征：文件名关键词
        (lambda name: any(kw in name for kw in ["方案", "流程", "步骤", "跟踪", "培养", "转正", "入职"]), 1, "文件名包含流程类关键词"),
    ],

    "form": [
        # 强特征：表单元素
        (lambda blocks: sum(1 for b in blocks if b.block_type == "table") >= 2, 3, "包含2个以上表格"),
        (lambda text: any(s in text for s in ["□", "☐", "☑", "√", "签字", "签名", "盖章"]), 3, "存在复选框/签字栏"),
        # 中特征：表单字段
        (lambda text: any(kw in text for kw in ["姓名", "部门", "日期", "岗位", "编号", "备注"]), 2, "包含表单常用字段"),
        # 弱特征：文件名关键词
        (lambda name: any(kw in name for kw in ["表", "表单", "报告", "复盘", "登记", "申请"]), 1, "文件名包含表单类关键词"),
    ],
}

# 人工复核阈值：最高分与次分差小于该值时，标记需要人工复核
_REVIEW_THRESHOLD = 2


# ==========================================================
# 三、内部工具函数
# ==========================================================

def _is_temporary_file(filename: str) -> bool:
    """判断是否为临时/无效文件，直接过滤"""
    return (
        filename.startswith("~$")
        or filename.startswith(".~")
        or filename.startswith(".")
        or filename.startswith("~")
    )


def _extract_full_text(blocks: List[DocumentBlock]) -> str:
    """提取全文文本用于特征匹配，解决只看前20块遗漏信息的问题"""
    return "\n".join([block.text for block in blocks if block.text])


# ==========================================================
# 四、核心评分逻辑
# ==========================================================

def _calculate_scores(
    file_path: str,
    blocks: List[DocumentBlock],
    full_text: str
) -> Dict[str, int]:
    """计算各类型得分，返回分数字典"""
    filename = os.path.basename(file_path).lower()
    scores = {doc_type: 0 for doc_type in _SCORING_RULES.keys()}

    for doc_type, rules in _SCORING_RULES.items():
        for rule_func, score, _ in rules:
            try:
                # 规则参数可能是文本、文件名或块列表，自动适配
                import inspect
                sig = inspect.signature(rule_func)
                param_count = len(sig.parameters)
                if param_count == 1:
                    # 尝试依次传入text、filename、blocks
                    try:
                        matched = rule_func(full_text)
                    except:
                        try:
                            matched = rule_func(filename)
                        except:
                            matched = rule_func(blocks)
                else:
                    matched = False
            except:
                matched = False

            if matched:
                scores[doc_type] += score

    return scores


def _collect_evidence(
    file_path: str,
    blocks: List[DocumentBlock],
    full_text: str
) -> List[str]:
    """收集所有命中的证据描述，用于调试追溯"""
    filename = os.path.basename(file_path).lower()
    evidence = []

    for doc_type, rules in _SCORING_RULES.items():
        for rule_func, _, desc in rules:
            try:
                import inspect
                sig = inspect.signature(rule_func)
                param_count = len(sig.parameters)
                if param_count == 1:
                    try:
                        matched = rule_func(full_text)
                    except:
                        try:
                            matched = rule_func(filename)
                        except:
                            matched = rule_func(blocks)
                else:
                    matched = False
            except:
                matched = False

            if matched:
                evidence.append(f"[{doc_type}] {desc}")

    return evidence


# ==========================================================
# 五、统一对外接口
# ==========================================================

def classify_document(file_path: str, blocks: List[DocumentBlock]) -> ClassificationResult:
    """
    文档分类主入口：证据评分制
    :param file_path: 文件完整路径
    :param blocks: 解析后的DocumentBlock列表
    :return: ClassificationResult 分类结果
    """
    filename = os.path.basename(file_path)

    # 第一步：先过滤临时文件
    if _is_temporary_file(filename):
        return ClassificationResult(
            primary_type="ignore",
            is_ignore=True,
            matched_evidence=["Word临时锁定文件/隐藏文件，直接跳过"]
        )

    # 第二步：提取全文特征
    full_text = _extract_full_text(blocks)
    if not full_text.strip():
        return ClassificationResult(
            primary_type="ignore",
            is_ignore=True,
            matched_evidence=["文档无有效文本内容"]
        )

    # 第三步：计算各类型得分
    scores = _calculate_scores(file_path, blocks, full_text)

    # 第四步：收集命中证据
    evidence = _collect_evidence(file_path, blocks, full_text)

    # 第五步：排序确定主类型和次类型
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_type = sorted_types[0][0]
    primary_score = sorted_types[0][1]

    # 次类型：得分≥主类型一半的都作为次要标签
    secondary_types = [
        t for t, s in sorted_types[1:]
        if s >= primary_score * 0.5 and s > 0
    ]

    # 第六步：判断是否需要人工复核
    needs_review = False
    if len(sorted_types) >= 2:
        second_score = sorted_types[1][1]
        if primary_score - second_score <= _REVIEW_THRESHOLD:
            needs_review = True
    if primary_score <= 1:
        needs_review = True  # 得分过低，可信度不足

    return ClassificationResult(
        primary_type=primary_type,
        secondary_types=secondary_types,
        scores=scores,
        matched_evidence=evidence,
        needs_review=needs_review,
        is_ignore=False
    )
