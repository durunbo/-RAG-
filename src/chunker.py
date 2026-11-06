from typing import List
from src.models import DocumentBlock, Chunk


class TextChunker:
    """
    通用文本切块器
    基于统一的 DocumentBlock 结构进行切块，不绑定任何文件格式
    核心规则：不跨章节边界、标题与正文绑定、短段落自动合并、超长内容按句子切分
    """

    def __init__(self, max_chunk_len: int = 500):
        """
        初始化切块器
        :param max_chunk_len: 单个切块的最大字符数，可根据 embedding 模型调整
        """
        self.max_chunk_len = max_chunk_len

    def chunk(self, blocks: List[DocumentBlock]) -> List[Chunk]:
        """
        将 DocumentBlock 列表切分为最终的 Chunk 列表
        :param blocks: 解析后的统一文档块列表
        :return: 切块结果列表
        """
        if not blocks:
            return []

        # 按原文顺序重新排序，确保后续处理顺序正确
        blocks = sorted(blocks, key=lambda x: x.order)

        chunks = []  # 最终输出的切块列表
        current_heading_path = []  # 当前所在的章节路径
        current_texts = []  # 当前累积待切分的文本片段
        current_len = 0  # 当前累积的总字符数
        chunk_index = 0  # 切块序号
        source_file = blocks[0].source_file  # 来源文件名

        def flush_chunks():
            """
            内部函数：将当前累积的文本生成切块并清空缓冲区
            长度未超限则直接输出一块；超长则按句子粒度拆分
            """
            nonlocal current_texts, current_len, chunk_index
            if not current_texts:
                return

            # 合并所有累积的文本
            full_text = "\n".join(current_texts)

            # 长度在阈值内，直接输出为一个块
            if current_len <= self.max_chunk_len:
                chunks.append(Chunk(
                    source_file=source_file,
                    heading_path=current_heading_path.copy(),
                    text=full_text,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            else:
                # 超过阈值，按句号拆分句子后再组块
                sentences = full_text.replace("。", "。\n").split("\n")
                temp_text = ""
                for sen in sentences:
                    if not sen.strip():
                        continue
                    # 当前句子加进去不超限就累加
                    if len(temp_text) + len(sen) <= self.max_chunk_len:
                        temp_text += sen
                    else:
                        # 超限就输出当前块，开启新块
                        if temp_text:
                            chunks.append(Chunk(
                                source_file=source_file,
                                heading_path=current_heading_path.copy(),
                                text=temp_text,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1
                        temp_text = sen
                # 处理最后剩余的文本
                if temp_text:
                    chunks.append(Chunk(
                        source_file=source_file,
                        heading_path=current_heading_path.copy(),
                        text=temp_text,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1

            # 清空缓冲区，准备下一轮累积
            current_texts = []
            current_len = 0

        # 遍历所有文档块，逐块处理
        for block in blocks:
            # 章节路径发生变化 → 先输出上一章节所有内容，绝不跨章节切块
            if block.heading_path != current_heading_path:
                flush_chunks()
                current_heading_path = block.heading_path.copy()

            # 将当前块文本加入缓冲区
            current_texts.append(block.text)
            current_len += len(block.text)

            # 累积长度达到阈值，触发一次切块输出
            if current_len >= self.max_chunk_len:
                flush_chunks()

        # 遍历结束，输出最后剩余的内容
        flush_chunks()
        return chunks
