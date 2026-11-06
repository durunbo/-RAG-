#分类型切块策略集
import re
from typing import List
from dataclasses import dataclass
from src.models import DocumentBlock, Chunk


# ==========================================================
# 扩展数据结构：兼容原有Chunk，新增业务元数据
# 不修改原有models.py，继承扩展
# ==========================================================
@dataclass
class EnhancedChunk(Chunk):
    """增强版切块，保留原有全部字段，新增业务元数据"""
    document_type: str = ""  # 文档类型：policy/faq/process/form
    source_page: int = None  # 来源页码
    section: str = ""  # 所属章节
    question: str = ""  # FAQ专属：对应问题
    stage: str = ""  # 流程专属：所属阶段
    responsible_department: str = ""  # 流程专属：责任部门
    form_name: str = ""  # 表单专属：表单名称


# ==========================================================
# 工具函数
# ==========================================================
# 「回答」标记：PDF 提取出来的答案正文本身已经带「A: 」前缀，
# 拼接时再加一次就会变成「A: A: 正文」，而这段文字会原样显示给用户。
_ANSWER_PREFIX_RE = re.compile(r'^\s*A\s*[：:]\s*', re.I)


def _strip_answer_prefix(text: str) -> str:
    """剥掉答案正文开头的「A: 」标记，只剥一层。

    为什么只剥一层：正常数据最多带一层；若答案正文碰巧以「A:」开头
    （例如解释某个代号），剥两层会误删正文。
    """
    return _ANSWER_PREFIX_RE.sub('', text, count=1)


def _split_by_sentence(text: str, max_len: int) -> List[str]:
    """按句子拆分长文本，优先完整句子"""
    sentences = re.split(r'(?<=[。！？；])', text)
    chunks = []
    current = ""
    for sen in sentences:
        if not sen.strip():
            continue
        if len(current) + len(sen) <= max_len:
            current += sen
        else:
            if current:
                chunks.append(current)
            current = sen
    if current:
        chunks.append(current)
    return chunks


def _is_article_title(text: str) -> bool:
    """严格匹配第X条，避免误判普通正文"""
    return bool(re.match(r'^\s*第[一二三四五六七八九十百千\d]+条\s*$', text.strip()))

def _is_chapter_title(text: str) -> bool:
    """严格匹配第X章"""
    return bool(re.match(r'^\s*第[一二三四五六七八九十百千\d]+章\s*$', text.strip()))


# ==========================================================
# 策略1：制度政策类（修复版）
# 修复：条款边界错误、标题单独成块、长条款不拆分、顺序问题
# ==========================================================
def chunk_policy(blocks: List[DocumentBlock], max_len: int = 600) -> List[EnhancedChunk]:
    """
    制度/规定类切块：章节+条款完整绑定
    正确逻辑：遇到新条款 → 先保存上一条 → 再开始新一条
    """
    if not blocks:
        return []

    # 按原文顺序排序
    blocks = sorted(blocks, key=lambda x: x.order)

    chunks = []
    current_heading_path = []
    current_texts = []
    current_len = 0
    chunk_index = 0
    source_file = blocks[0].source_file
    current_page = None

    def flush():
        nonlocal current_texts, current_len, chunk_index, current_page
        if not current_texts:
            return

        full_text = "\n".join(current_texts)

        # 单条超长则按句子拆分
        if len(full_text) <= max_len:
            chunks.append(EnhancedChunk(
                source_file=source_file,
                heading_path=current_heading_path.copy(),
                text=full_text,
                chunk_index=chunk_index,
                document_type="policy",
                source_page=current_page,
                section=current_heading_path[-1] if current_heading_path else ""
            ))
            chunk_index += 1
        else:
            parts = _split_by_sentence(full_text, max_len)
            for part in parts:
                chunks.append(EnhancedChunk(
                    source_file=source_file,
                    heading_path=current_heading_path.copy(),
                    text=part,
                    chunk_index=chunk_index,
                    document_type="policy",
                    source_page=current_page,
                    section=current_heading_path[-1] if current_heading_path else ""
                ))
                chunk_index += 1

        current_texts = []
        current_len = 0

    for block in blocks:
        if block.page:
            current_page = block.page

        # ========== 核心：优先用结构判断边界 ==========
        if block.block_type == "heading":
            # 层级变了 → 先切分上一段
            if block.heading_path != current_heading_path:
                flush()
                current_heading_path = block.heading_path.copy()

            # 标题文本加入当前块，和正文绑定，不单独成块
            current_texts.append(block.text.strip())
            current_len += len(block.text)
            continue

        # ========== 兜底：纯文本正则识别（仅针对无样式文档） ==========
        # 只有当 heading_path 为空、且文本匹配条款格式时，才触发正则切分
        if not current_heading_path and _is_article_title(block.text):
            flush()
            current_heading_path = [block.text.strip()]
            current_texts.append(block.text.strip())
            current_len = len(block.text)
            continue

        # ========== 普通正文 ==========
        current_texts.append(block.text.strip())
        current_len += len(block.text)

        if current_len >= max_len:
            flush()

    # 收尾
    flush()
    return chunks


# ==========================================================
# 策略2：FAQ问答类（修复版）
# 修复：丢问题、分类标题误判、块内多问答无法识别、长答案不拆分
# ==========================================================
def chunk_faq(blocks: List[DocumentBlock], max_len: int = 500) -> List[EnhancedChunk]:
    """
    FAQ类切块：完整问答对为单位，支持块内多问答拆分
    不丢失无答案问题，长答案拆分后保留问题前缀
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda x: x.order)
    chunks = []
    chunk_index = 0
    source_file = blocks[0].source_file

    # ========== 问题识别规则 ==========
    # 1. 明确标记型
    _QA_MARK_RE = re.compile(r'^\s*(Q[：:]|问题[：:]|问[：:]|FAQ[：:])\s*(.+)', re.I)
    # 2. 编号问句型
    _NUM_QUESTION_RE = re.compile(r'^\s*(\d+\.|[一二三四五六七八九十]+、)\s*(.+\？)\s*$')
    # 3. 自然问句疑问词白名单（必须命中至少一个，大幅降低误判）
    _QUESTION_WORDS = {
        "如何", "怎么", "怎么办", "为啥", "为什么", "为何",
        "是否", "能不能", "可以吗",
        "哪里", "在哪", "什么", "哪些", "哪个", "谁",
        "什么时候", "多久", "多少", "几天", "几年"
    }

    def _is_natural_question(line: str, is_block_start: bool) -> bool:
        """严格自然问句：段首 + 问号结尾 + 长度适中 + 含疑问词 + 排除说明句"""
        line = line.strip()
        # 基础条件
        if not line.endswith("？"):
            return False
        if len(line) < 8 or len(line) > 60:
            return False
        if not is_block_start:
            return False
        # 必须包含疑问词
        has_question_word = any(kw in line for kw in _QUESTION_WORDS)
        if not has_question_word:
            return False
        # 排除明显的说明/举例句
        exclude_keywords = ["如下", "说明", "是指", "如下所示", "注意", "例如", "比如"]
        for kw in exclude_keywords:
            if kw in line:
                return False
        return True

    def _is_new_question(line: str, is_block_start: bool) -> tuple[bool, str]:
        """判断是否为新问题，三级优先级：明确标记 > 编号问句 > 严格自然问句"""
        line = line.strip()
        if not line:
            return False, ""

        # 优先级1：明确标记
        mark_match = _QA_MARK_RE.match(line)
        if mark_match:
            return True, mark_match.group(2).strip()

        # 优先级2：编号问句
        num_match = _NUM_QUESTION_RE.match(line)
        if num_match:
            return True, num_match.group(2).strip()

        # 优先级3：自然问句兜底
        if _is_natural_question(line, is_block_start):
            return True, line.strip()

        return False, ""

    # ========== 超长拆分：三级兜底，问题优先 ==========
    def _split_long_answer(question: str, answer: str, max_len: int) -> List[str]:
        """
        超长答案拆分：段落→句子→硬拆 三级兜底
        边界策略：可用空间过小时，优先保证问题完整，答案尽量承载
        """
        prefix = f"Q: {question}\nA: "
        available = max_len - len(prefix) - 5  # 预留缓冲

        # 边界兜底：可用空间过小，问题优先，答案仅截取可容纳部分
        if available < 30:
            return [answer[:available]]

        result_parts = []

        # 第一级：按段落拆分
        paragraphs = [p.strip() for p in answer.split('\n') if p.strip()]
        temp = ""
        for para in paragraphs:
            if len(temp) + len(para) <= available:
                temp += para + "\n"
            else:
                if temp.strip():
                    result_parts.append(temp.strip())
                # 单段就超长，进入下一级拆分
                if len(para) > available:
                    # 第二级：按句子拆分
                    sentences = re.split(r'(?<=[。！？；])', para)
                    sub_temp = ""
                    for sen in sentences:
                        if not sen.strip():
                            continue
                        if len(sub_temp) + len(sen) <= available:
                            sub_temp += sen
                        else:
                            if sub_temp:
                                result_parts.append(sub_temp)
                            # 第三级：单句仍超长，按字符硬拆兜底
                            if len(sen) > available:
                                for i in range(0, len(sen), available):
                                    result_parts.append(sen[i:i + available])
                            else:
                                sub_temp = sen
                    if sub_temp:
                        result_parts.append(sub_temp)
                else:
                    temp = para + "\n"
        if temp.strip():
            result_parts.append(temp.strip())

        return result_parts

    # ========== 收尾生成Chunk ==========
    def _save_qa(question: str, answer_lines: List[str], section: str, heading_path: List[str], start_page: int):
        nonlocal chunk_index
        if not question.strip():
            return

        answer = _strip_answer_prefix("\n".join(answer_lines).strip())
        full_text = f"Q: {question}\nA: {answer}" if answer else f"Q: {question}\nA: （暂无答案说明）"

        # 长度正常 → 单块
        if len(full_text) <= max_len or not answer:
            chunks.append(EnhancedChunk(
                source_file=source_file,
                heading_path=heading_path.copy(),
                text=full_text,
                chunk_index=chunk_index,
                document_type="faq",
                source_page=start_page,  # 问答起始页
                question=question,
                section=section
            ))
            chunk_index += 1
            return

        # 超长 → 拆分后每块都带完整问题
        answer_parts = _split_long_answer(question, answer, max_len)
        for part in answer_parts:
            chunk_text = f"Q: {question}\nA: {part}"
            chunks.append(EnhancedChunk(
                source_file=source_file,
                heading_path=heading_path.copy(),
                text=chunk_text,
                chunk_index=chunk_index,
                document_type="faq",
                source_page=start_page,
                question=question,
                section=section
            ))
            chunk_index += 1

    # ========== 主逻辑：按DocumentBlock顺序遍历 ==========
    current_question = ""
    current_answer = []
    current_section = ""
    current_heading_path = []
    current_start_page = None

    for block in blocks:
        # 【修正1】heading_path 只更新上下文，不截断问答
        # 章节变化仅更新section和元数据，问答继续收集，直到遇到下一个明确问题
        if block.heading_path and block.heading_path != current_heading_path:
            current_heading_path = block.heading_path.copy()
            current_section = block.heading_path[-1] if block.heading_path else ""

        # 表格块直接归入当前答案
        if block.block_type == "table":
            if current_question:
                current_answer.append(block.text)
            continue

        # 段落块逐行检查
        lines = block.text.split('\n')
        for line_idx, line in enumerate(lines):
            is_block_start = (line_idx == 0)
            is_q, q_text = _is_new_question(line, is_block_start)

            if is_q:
                # 只有遇到明确新问题，才结束上一个问答
                if current_question:
                    _save_qa(current_question, current_answer, current_section, current_heading_path,
                             current_start_page)
                # 开启新问答
                current_question = q_text
                current_answer = []
                current_start_page = block.page
                continue

            # 普通行：属于当前答案
            if current_question and line.strip():
                current_answer.append(line.strip())

    # 收尾最后一个
    if current_question:
        _save_qa(current_question, current_answer, current_section, current_heading_path, current_start_page)

    return chunks




# ==========================================================
# 策略3：流程类文档切块（入职/转正/培养方案等带时间节点的文档）
# 核心原则：以「阶段/时间节点」为单位切块，保证每个Chunk自带stage标签
# 输出扩展字段：stage（时间/阶段标签）、responsible_department（责任部门）
# 重要定位：stage仅记录阶段边界标题，不等同于Chunk内全部时间信息
# 后续若需精细时间检索，需单独从Chunk正文中解析提取
# ==========================================================

# 时间节点标题正则：用于判定阶段边界，优先级高于步骤标记
# 匹配范围：入职X天/周、第X天/周、X个月、半年/半个月等典型流程节点
# 设计原则：严格限定前缀场景，避免把正文里的普通时间描述误判为阶段标题
# 暂不盲目扩充匹配范围，等拿到真实文档标题再评估
_STAGE_TIME_RE = re.compile(
    r"(?:"
    r"入职\s*[0-9一二三四五六七八九十两]+\s*[天周]"
    r"|第\s*[一二三四五六七八九十两0-9]+\s*[天周]"
    r"|[一二三四五六七八九十两0-9]+\s*个?月"
    r"|半年|半个月"
    r")"
)

# 步骤/阶段标记兜底：当文档完全没有时间型阶段标题时，用「第X步/阶段」作为阶段边界
# 仅兜底使用，优先级低于时间型标题
_STAGE_STEP_RE = re.compile(r"第\s*[一二三四五六七八九十0-9]+\s*(?:步|阶段)")

# 责任部门识别正则：仅匹配部门类显式标注
# 修复：移除「责任人」匹配，避免匹配到人名导致字段语义错误
# 仅匹配部门级主体，识别不到返回空，宁空勿错
_DEPT_RE = re.compile(
    r"(?:责任部门|负责部门|经办部门|牵头部门|主管部门)\s*[：:]\s*([^\s，。,；;]+)"
)


def _extract_time_stage(text: str) -> str:
    """提取时间型阶段标签，仅匹配开头为时间阶段格式的标题"""
    text = text.strip()
    m = re.match(r'^\s*' + _STAGE_TIME_RE.pattern, text)
    if m:
        return m.group(0).strip()
    return ""


def _extract_step_stage(text: str) -> str:
    """提取步骤型阶段标签，仅匹配开头为第X步/阶段的标题"""
    text = text.strip()
    m = re.match(r'^\s*' + _STAGE_STEP_RE.pattern, text)
    if m:
        return m.group(0).strip()
    return ""


def _extract_department(text: str) -> str:
    """
    从正文提取责任部门信息
    仅匹配部门类显式标注，不匹配个人姓名
    识别不到返回空字符串，宁空勿错
    """
    m = _DEPT_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def chunk_process(blocks: List[DocumentBlock], max_len: int = 500) -> List[EnhancedChunk]:
    """
    流程类文档切块：一个阶段（时间节点）对应一个Chunk
    核心设计：
    1. 时间型阶段标题优先级最高，存在时步骤标记不触发阶段切换
    2. 每个 Chunk 完整继承对应阶段的 stage、责任部门、章节路径
    3. 阶段内子标题不重置阶段信息，跨出阶段分支自动清空
    4. 超长拆分后所有子块复用同一阶段元数据，避免时间标签丢失

    已确认修复：
    - 起始页码补全：首个有效页码自动作为Chunk起始页，标题无页码不丢失
    - 阶段跨分支残留：按heading_path前缀判断归属，跨出分支自动清空阶段/部门
    - 责任人语义错误：仅匹配部门类关键词
    - 步骤兜底优先级：预扫描全文是否有时间型标题，统一决定是否启用兜底
    - 超长单句溢出：句子拆分后增加字符级硬拆兜底
    """
    if not blocks:
        return []

    if max_len <= 0:
        raise ValueError(f"max_len 必须大于 0，当前值：{max_len}")

    blocks = sorted(blocks, key=lambda x: x.order)
    chunks = []
    chunk_index = 0
    source_file = blocks[0].source_file

    current_heading_path = []
    current_stage = ""       # 当前阶段标签，仅阶段标题触发更新
    current_stage_path = []  # 当前阶段标题对应的完整层级路径，用于判断子标题归属
    current_department = ""  # 当前阶段责任部门，仅阶段标题切换时重置
    current_texts = []
    current_len = 0
    current_chunk_start_page = None  # 当前累积Chunk的起始页码

    # ========== 预扫描：判断全文是否存在【时间型】阶段标题 ==========
    # 仅时间型标题才算有阶段结构；步骤型不算，避免动态切换兜底开关
    has_time_stage = False
    for block in blocks:
        if block.block_type == "heading":
            if _extract_time_stage(block.text):
                has_time_stage = True
                break

    def _make_chunk(text: str, page, heading_path, stage, dept, section):
        """内部辅助：统一构建EnhancedChunk，减少重复代码"""
        nonlocal chunk_index

        chunk = EnhancedChunk(
            source_file=source_file,
            heading_path=heading_path.copy(),
            text=text,
            chunk_index=chunk_index,
            document_type="process",
            source_page=page,
            stage=stage,
            responsible_department=dept,
            section=section,
        )

        chunk_index += 1
        return chunk

    def flush():
        nonlocal current_texts, current_len, current_chunk_start_page

        if not current_texts:
            return

        full_text = "\n".join(current_texts)
        heading_path = current_heading_path.copy()
        stage = current_stage
        dept = current_department
        page = current_chunk_start_page
        section = current_heading_path[-1] if current_heading_path else ""

        # 长度正常：单块输出
        if len(full_text) <= max_len:
            chunks.append(
                _make_chunk(full_text, page, heading_path, stage, dept, section)
            )

            current_texts = []
            current_len = 0
            current_chunk_start_page = None
            return

        # 超长：先按句子拆分
        parts = _split_by_sentence(full_text, max_len)
        final_parts = []

        for part in parts:
            # 超长单句兜底：单句仍超限时按字符级硬拆，确保不超max_len
            if len(part) <= max_len:
                final_parts.append(part)
            else:
                for i in range(0, len(part), max_len):
                    final_parts.append(part[i:i + max_len])

        # 每个子块复用同一阶段元数据
        for part in final_parts:
            chunks.append(
                _make_chunk(part, page, heading_path, stage, dept, section)
            )

        current_texts = []
        current_len = 0
        current_chunk_start_page = None

    for block in blocks:
        block_page = block.page

        # ========== 标题块：判断是否为阶段边界 ==========
        if block.block_type == "heading":

            # 优先判断时间型阶段标题
            time_stage = _extract_time_stage(block.text)

            if time_stage:
                # 先收尾上一阶段，防止新标题页码被分配给上一块
                flush()

                # 开启新阶段，并在新Chunk上记录标题页码
                current_heading_path = block.heading_path.copy()
                current_stage = time_stage
                current_stage_path = block.heading_path.copy()
                current_department = ""
                current_chunk_start_page = block_page

                current_texts.append(block.text.strip())
                current_len = len(block.text)
                continue

            # 有时间型阶段标题时，忽略步骤型标题，不触发阶段切换
            if not has_time_stage:
                step_stage = _extract_step_stage(block.text)

                if step_stage:
                    # 先收尾上一阶段，再初始化新阶段
                    flush()

                    current_heading_path = block.heading_path.copy()
                    current_stage = step_stage
                    current_stage_path = block.heading_path.copy()
                    current_department = ""
                    current_chunk_start_page = block_page

                    current_texts.append(block.text.strip())
                    current_len = len(block.text)
                    continue

            # 普通子标题：更新路径，按前缀判断是否重置阶段信息
            if block.heading_path != current_heading_path:
                # 必须先保存旧Chunk，再更新新Chunk的路径和页码
                flush()

                # 新标题路径如果不属于当前阶段路径的子层级，
                # 说明已跨出阶段分支，清空旧阶段和责任部门
                is_in_current_stage = False

                if current_stage and current_stage_path:
                    prefix_len = len(current_stage_path)

                    if len(block.heading_path) >= prefix_len:
                        is_in_current_stage = (
                            block.heading_path[:prefix_len] == current_stage_path
                        )

                if not is_in_current_stage:
                    current_stage = ""
                    current_stage_path = []
                    current_department = ""

                current_heading_path = block.heading_path.copy()

            # 当前Chunk还没有确定起始页码时，使用首个有效页码
            # 不覆盖已经确定的起始页码
            if current_chunk_start_page is None and block_page is not None:
                current_chunk_start_page = block_page

            current_texts.append(block.text.strip())
            current_len += len(block.text)
            continue

        # ========== 兜底：全文无时间型阶段标题时，用「第X步」作为阶段边界 ==========
        if not has_time_stage:
            step_label = _extract_step_stage(block.text)

            if step_label:
                # 先收尾旧Chunk，再使用当前正文块的真实heading_path
                flush()

                # 不把整段正文伪装成层级路径，保留DocumentBlock原始结构
                current_heading_path = block.heading_path.copy()
                current_stage = step_label
                current_stage_path = block.heading_path.copy()
                current_department = ""
                current_chunk_start_page = block_page

                current_texts.append(block.text.strip())
                current_len = len(block.text)
                continue

        # ========== 普通正文：累积文本 + 尝试提取责任部门 ==========

        # 当前Chunk仍未确定起始页码时，取首个有效页码
        # 特别处理阶段标题页码为空、正文页码有效的情况
        if current_chunk_start_page is None and block_page is not None:
            current_chunk_start_page = block_page

        current_texts.append(block.text.strip())
        current_len += len(block.text)

        # 责任部门只提取一次，找到后不再重复识别
        if not current_department:
            current_department = _extract_department(block.text)

        if current_len >= max_len:
            flush()

    # 收尾最后一个Chunk
    flush()
    return chunks



# ==========================================================
# 策略4：表单类（新增专属策略，不再用通用切块）（目前未出现，保留策略，待后续扩展）
# 按表格为单位切块，保留表名、字段结构、填写说明
# ==========================================================
def chunk_form(blocks: List[DocumentBlock], max_len: int = 800) -> List[EnhancedChunk]:
    """
    表单类文档切块：以单个表格为核心单位
    每个表格绑定最近的表名标题和填写说明，不强行按字符切碎表格
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda x: x.order)
    chunks = []
    chunk_index = 0
    source_file = blocks[0].source_file
    current_form_name = ""
    current_heading_path = []
    current_page = None
    pending_desc = []  # 表格前的说明文字

    for block in blocks:
        if block.page:
            current_page = block.page

        # 标题：更新表名候选
        if block.block_type == "heading":
            current_heading_path = block.heading_path.copy()
            current_form_name = block.text.strip()
            pending_desc = []
            continue

        # 普通段落：作为表格说明暂存
        if block.block_type == "paragraph":
            pending_desc.append(block.text.strip())
            continue

        # 表格：输出一个完整块
        if block.block_type == "table":
            form_text_parts = []
            if current_form_name:
                form_text_parts.append(f"【表单名称】{current_form_name}")
            if pending_desc:
                form_text_parts.append("【填写说明】" + "\n".join(pending_desc))
            form_text_parts.append("【表格内容】")
            form_text_parts.append(block.text.strip())

            full_text = "\n".join(form_text_parts)

            # 超长表格只做行级拆分，保留表头
            if len(full_text) <= max_len:
                chunks.append(EnhancedChunk(
                    source_file=source_file,
                    heading_path=current_heading_path.copy(),
                    text=full_text,
                    chunk_index=chunk_index,
                    document_type="form",
                    source_page=current_page,
                    form_name=current_form_name
                ))
                chunk_index += 1
            else:
                # 表格拆分：保留表头，拆分数据行
                lines = block.text.strip().split("\n")
                header = lines[0] if lines else ""
                current_table = [header]
                current_len = len(header)

                for line in lines[1:]:
                    if current_len + len(line) <= max_len - 50:
                        current_table.append(line)
                        current_len += len(line)
                    else:
                        table_text = "\n".join(current_table)
                        chunk_text = f"【表单名称】{current_form_name}\n【表格内容】\n{table_text}"
                        chunks.append(EnhancedChunk(
                            source_file=source_file,
                            heading_path=current_heading_path.copy(),
                            text=chunk_text,
                            chunk_index=chunk_index,
                            document_type="form",
                            source_page=current_page,
                            form_name=current_form_name
                        ))
                        chunk_index += 1
                        current_table = [header, line]
                        current_len = len(header) + len(line)

                if len(current_table) > 1:
                    table_text = "\n".join(current_table)
                    chunk_text = f"【表单名称】{current_form_name}\n【表格内容】\n{table_text}"
                    chunks.append(EnhancedChunk(
                        source_file=source_file,
                        heading_path=current_heading_path.copy(),
                        text=chunk_text,
                        chunk_index=chunk_index,
                        document_type="form",
                        source_page=current_page,
                        form_name=current_form_name
                    ))
                    chunk_index += 1

            pending_desc = []

    return chunks


# ==========================================================
# 统一调度入口（严格版）
# 修复：不悄悄降级为通用切块，未知类型提示人工确认
# ==========================================================
def get_chunks_by_type(
        doc_type: str,
        blocks: List[DocumentBlock],
        max_len: int = 500
) -> List[EnhancedChunk]:
    """
    按文档类型调度对应切块策略
    严格匹配：未知类型抛出提示，不自动用通用切块兜底
    """
    if doc_type == "policy":
        return chunk_policy(blocks, max_len)
    elif doc_type == "faq":
        return chunk_faq(blocks, max_len)
    elif doc_type == "process":
        return chunk_process(blocks, max_len)
    elif doc_type == "form":
        return chunk_form(blocks, max_len)
    elif doc_type == "general":
        # 通用类型兜底，复用原有通用切块逻辑
        from src.chunker import TextChunker
        chunker = TextChunker(max_chunk_len=max_len)
        basic_chunks = chunker.chunk(blocks)
        # 转换为增强格式
        return [
            EnhancedChunk(**c.__dict__, document_type="general")
            for c in basic_chunks
        ]
    else:
        raise ValueError(
            f"未知文档类型：{doc_type}，暂不支持自动切块，建议人工确认后选择对应策略"
        )
