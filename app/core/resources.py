import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.mock_token_auth import MockTokenAuthenticator, parse_mock_tokens
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.services.document_tasks import IngestTaskStore
from app.services.embedding_client import EmbeddingBackend, create_embedding_backend
from app.services.llm_client import LLMClient, create_llm_client
from app.services.rag_service import RAGConfig, RAGService
from src.session.session_manager import SessionManager
from src.vector_store.vector_store import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class AppResources:
    """
    保存微服务进程共享的核心资源。

    所有请求共用同一个 embedding 后端、向量库、SessionManager 和 LLM 客户端实例，
    避免每个请求重复创建模型对象或 HTTP 客户端。

    字段类型刻意用接口（EmbeddingBackend / LLMClient）而不是具体实现，
    这样换 provider 时这里不需要改。
    """

    embedding_model: EmbeddingBackend
    vector_store: VectorStore
    session_manager: SessionManager
    llm_client: LLMClient
    rag_service: RAGService
    db_engine: AsyncEngine  # 共享的数据库引擎（用于清理时关闭连接池）
    # token 认证器：无状态，只持有模拟 token 映射表，可跨请求复用
    token_auth: MockTokenAuthenticator
    # 文档入库任务表。必须跨请求共享，否则轮询时看不到上传时建的任务
    ingest_task_store: IngestTaskStore
    # 允许使用文档管理功能的 user_id 集合。空集合 = 功能整体关闭
    admin_user_ids: frozenset[str] = frozenset()
    allowed_upload_extensions: frozenset[str] = frozenset({".pdf", ".docx"})
    upload_max_bytes: int = 20 * 1024 * 1024
    upload_temp_dir: Path = PROJECT_ROOT / "data" / "uploads"

    def is_admin(self, user_id: str) -> bool:
        """是否为文档管理员。名单为空时对任何人都是 False。"""
        return bool(user_id) and user_id in self.admin_user_ids


def create_app_resources(
    settings: Settings | None = None,
) -> AppResources:
    """
    根据统一配置初始化应用资源。

    注意：
    1. Embedding 后端由 embedding_provider 决定，当前为本地 BGE-M3，
       创建实例时不会立即加载模型（延迟到访问 dimension）。
    2. VectorStore 使用 PostgreSQL + pgvector，初始化时创建连接池。
    3. SessionManager 复用 VectorStore 的数据库引擎，统一管理连接池。
    4. LLM 客户端由 llm_provider 决定（本地 Ollama 或云端 OpenAI 兼容服务），
       无状态，初始化成本极低。
    """
    active_settings = settings or get_settings()

    # 一、初始化向量化后端
    # 具体实现与设备、批大小都由配置决定，这里不再直接 import EmbeddingModel
    embedding_model = create_embedding_backend(active_settings)

    # 二、初始化 PostgreSQL 向量库
    vector_store_config = VectorStoreConfig(
        database_url=str(active_settings.database_url),
    )

    vector_store = VectorStore(
        embedding_model=embedding_model,
        config=vector_store_config,
    )

    # 三、创建共享的数据库引擎（供 SessionManager 复用）
    db_engine = create_async_engine(str(active_settings.database_url))

    # 四、初始化 SessionManager（复用 engine）
    session_manager = SessionManager(engine=db_engine)

    # 五、认证与限流
    #
    # 测试阶段用模拟认证：token 精确匹配表，不解析、不验签、不碰数据库。
    # 三项前置校验，刻意让服务启动失败而不是运行时才发现：
    #   1. auth_provider 只允许 mock，写了未实现的值直接报错；
    #   2. 生产环境禁止 mock —— 防止「忘了接真实认证」就上线；
    #   3. mock_tokens 必须非空 —— 否则任何人都无法通过认证。
    if active_settings.auth_provider != "mock":
        raise ValueError(
            f"未实现的认证 provider: {active_settings.auth_provider!r}"
        )
    if active_settings.app_env == "production":
        raise ValueError(
            "生产环境禁止使用模拟 Token 认证（auth_provider=mock）"
        )
    mock_token_map = parse_mock_tokens(active_settings.mock_tokens)
    if not mock_token_map:
        raise ValueError(
            "mock_tokens 未配置：认证无法工作，服务拒绝启动"
        )
    token_auth = MockTokenAuthenticator(token_map=mock_token_map)

    # 六之一、文档管理
    #
    # 管理员名单为空时功能整体关闭，但这不影响启动：
    # 接口一律按「不存在」处理，日志里必须留一条，否则会以为是接口没写好。
    admin_user_ids = active_settings.admin_user_id_set
    if not admin_user_ids:
        logger.warning(
            "文档管理功能未启用：RAG_ADMIN_USER_IDS 为空，"
            "上传/查询/删除接口一律按「不存在」处理"
        )
    upload_temp_dir = Path(active_settings.upload_temp_dir)
    upload_temp_dir.mkdir(parents=True, exist_ok=True)

    ingest_task_store = IngestTaskStore(
        ttl_minutes=active_settings.ingest_task_ttl_minutes,
        max_records=active_settings.ingest_task_max_records,
    )

    # 六、初始化 LLM 客户端（provider 决定是本地 Ollama 还是云端 OpenAI 兼容服务）
    llm_client = create_llm_client(active_settings)

    # 七、初始化 RAG 服务（无状态，复用上面的 VectorStore 与 LLM 客户端）
    rag_service = RAGService(
        vector_store=vector_store,
        llm_client=llm_client,
        config=RAGConfig(
            max_concurrency=active_settings.llm_max_concurrency,
            min_score=active_settings.retrieval_min_score,
        ),
    )

    # 八、统一返回，供 FastAPI 生命周期管理
    return AppResources(
        embedding_model=embedding_model,
        vector_store=vector_store,
        session_manager=session_manager,
        llm_client=llm_client,
        rag_service=rag_service,
        db_engine=db_engine,
        token_auth=token_auth,
        ingest_task_store=ingest_task_store,
        admin_user_ids=admin_user_ids,
        allowed_upload_extensions=active_settings.allowed_upload_extensions,
        upload_max_bytes=active_settings.upload_max_bytes,
        upload_temp_dir=upload_temp_dir,
    )
