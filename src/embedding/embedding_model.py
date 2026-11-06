from typing import List, Optional
from dataclasses import dataclass

from src.chunk_strategies import EnhancedChunk


# ==========================================================
# 配置类
# ==========================================================
@dataclass
class EmbeddingConfig:
    """
    Embedding 配置
    默认使用本地模型路径，避免无网环境下自动请求 HuggingFace
    可根据实际模型存放位置修改 model_name_or_path
    """
    model_name_or_path: str = r"D:\AI\Models\bge-m3"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32


# ==========================================================
# 工具函数：构造 Embedding 输入文本
# 规则完全对齐定版：heading_path 拼接 + 换行 + 正文
# ==========================================================
def build_embedding_text(chunk: EnhancedChunk) -> str:
    """
    根据 EnhancedChunk 生成 Embedding 输入文本
    严格遵守：仅 heading_path + text 参与向量化，所有元数据字段不进入
    """
    heading_part = ""
    if chunk.heading_path:
        heading_part = " / ".join(chunk.heading_path)

    body = chunk.text if chunk.text else ""

    if heading_part:
        return f"{heading_part}\n{body}"
    else:
        return body


# ==========================================================
# 核心模型类
# ==========================================================
class EmbeddingModel:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化仅加载配置，模型延迟加载
        不创建实例就不加载模型，支持轻量引用
        """
        self.config = config or EmbeddingConfig()
        self._model = None  # 模型延迟加载
        self._dim: Optional[int] = None  # 维度缓存

    def _load_model(self):
        """
        延迟加载模型：第一次调用 encode 时才真正加载
        全局只加载一次，后续调用复用
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                model_name_or_path=self.config.model_name_or_path,
                device=self.config.device
            )
            # 缓存维度
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """向量维度，缓存结果，不重复计算"""
        if self._dim is None:
            self._load_model()
        return self._dim

    def encode(self, chunk: EnhancedChunk) -> List[float]:
        """
        单个 Chunk 生成向量
        空文本返回对应维度零向量，严格保持位置对应
        """
        text = build_embedding_text(chunk)

        # 空文本处理：返回零向量占位，不跳过，保证下标对应
        if not text.strip():
            return [0.0] * self.dimension

        model = self._load_model()
        embedding = model.encode(
            sentences=[text],
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=1
        )[0].tolist()

        return embedding

    def encode_batch(self, chunks: List[EnhancedChunk]) -> List[List[float]]:
        """
        批量生成向量（真正批量推理，不是循环单条）
        核心保证：输出列表下标与输入 chunks 严格一一对应
        空文本位置返回零向量，不跳过、不错位
        """
        if not chunks:
            return []

        # 批量构造文本，顺序和 chunks 完全对应
        texts = [build_embedding_text(chunk) for chunk in chunks]

        # 记录空文本索引，后续回填零向量
        empty_indices = [i for i, t in enumerate(texts) if not t.strip()]
        non_empty_texts = [t for t in texts if t.strip()]

        # 全部为空则直接返回全零向量
        if not non_empty_texts:
            return [[0.0] * self.dimension for _ in chunks]

        model = self._load_model()

        # 真正批量推理
        embeddings = model.encode(
            sentences=non_empty_texts,
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=self.config.batch_size
        ).tolist()

        # 回填空文本位置的零向量，恢复原始顺序
        result = []
        emb_idx = 0
        for i in range(len(chunks)):
            if i in empty_indices:
                result.append([0.0] * self.dimension)
            else:
                result.append(embeddings[emb_idx])
                emb_idx += 1

        return result

    def encode_query(self, query: str) -> List[float]:
        """
        用户查询向量化
        独立接口，与文档向量化解耦，后续可单独调整查询构造策略
        不依赖 EnhancedChunk，纯字符串输入
        """
        text = query.strip()
        # 空查询返回零向量
        if not text:
            return [0.0] * self.dimension

        model = self._load_model()
        embedding = model.encode(
            sentences=[text],
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=1
        )[0].tolist()

        return embedding

    def get_metadata(self) -> dict:
        """输出嵌入元数据，用于写入向量库，保证建库检索配置一致"""
        return {
            "embedding_model": self.config.model_name_or_path,
            "vector_dim": self.dimension,
            "normalize_embeddings": self.config.normalize_embeddings,
            "device": self.config.device,
            "batch_size": self.config.batch_size
        }
