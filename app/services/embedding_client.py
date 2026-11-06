"""Embedding 客户端统一契约与工厂。

与 LLM 的选择彼此独立：llm_provider 换成云端不影响向量化，反之亦然。

边界说明（重要）：
换 embedding 模型不是「改个配置就完事」。即使两个模型输出维度相同，向量空间
也可能不同——用旧模型生成的库去检索新模型编码的 query，不会报错，只会安静地
返回错误结果。因此切换 embedding 后必须重新 ingest + 重建索引。
LLM 没有这个问题，换生成模型只影响回答，检索与向量库完全不动。

当前只有 local 一个实现。刻意不预造云端实现的空壳：没有真实需求时，
空壳只会带来「看起来支持、实际没验证过」的错觉。
"""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """向量化后端契约。

    只声明 VectorStore 与启动流程实际用到的成员（可用 grep 验证：
    vector_store.py 用 encode_batch / encode_query / get_metadata，
    main.py 与 health.py 用 dimension）。
    单条 encode() 不进来：云端 Embedding API 通常只提供批量接口，
    把它写进契约会迫使我们为不存在的实现提供兼容层。
    """

    @property
    def dimension(self) -> int:
        """向量维度。启动时用它校验库内向量列是否匹配。"""
        ...

    def encode_batch(self, chunks: list) -> list[list[float]]:
        """批量编码文档切片。"""
        ...

    def encode_query(self, query: str) -> list[float]:
        """编码检索查询。"""
        ...

    def get_metadata(self) -> dict:
        """返回写入向量库元数据的模型信息（名称、维度、设备等）。"""
        ...


def create_embedding_backend(settings: Settings) -> EmbeddingBackend:
    """按 settings.embedding_provider 创建向量化后端。

    新增实现（例如云端 Embedding）的步骤：
    1. config.py 的 embedding_provider Literal 加取值；
    2. 新建适配器，实现 dimension / encode_batch / encode_query / get_metadata；
    3. 在下面加一个 elif 分支。
    注意还要处理「换模型 = 重建索引」的迁移流程，不能只切配置。
    """
    provider = (settings.embedding_provider or "").strip().lower()

    if provider == "local":
        # 函数内导入：src 层是文档处理侧，app 启动时才需要加载，
        # 顶层导入会让「只跑脚本」的场景也无谓依赖 sentence-transformers
        from src.embedding.embedding_model import EmbeddingConfig, EmbeddingModel

        # 先校验再构造：校验失败就没必要创建实例。
        # 在这里直接失败，而不是「启动成功、/health/ready 永久 503」。
        # 与 /health/ready 的 is_dir 检查口径一致：都只认「本地目录」这一种输入，
        # 不再声称支持 HuggingFace 模型标识（那会触发联网下载，离线环境必挂）。
        #
        # 选 fail-fast 而不是告警的理由：目录缺失属于配置或挂载错误，重试不会自愈。
        # 让容器带着一个永远不可用的实例跑着，比启动即失败更难排查。
        model_path = Path(str(settings.embedding_model_path))
        if not model_path.is_dir():
            raise ValueError(
                f"Embedding 模型目录不存在：{model_path}。"
                "本服务只支持本地模型目录（不支持 HuggingFace 模型标识），"
                "请检查 RAG_EMBEDDING_MODEL_PATH 或容器挂载是否正确"
            )

        backend: EmbeddingBackend = EmbeddingModel(
            config=EmbeddingConfig(
                model_name_or_path=str(settings.embedding_model_path),
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            )
        )

    else:
        raise ValueError(
            f"不支持的 RAG_EMBEDDING_PROVIDER: {provider!r}，"
            "当前已实现：local"
        )

    if not isinstance(backend, EmbeddingBackend):
        raise TypeError(
            f"{type(backend).__name__} 未实现 EmbeddingBackend 契约"
        )

    logger.info(
        "Embedding 后端就绪：provider=%s model=%s device=%s batch_size=%s",
        provider,
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
    return backend
