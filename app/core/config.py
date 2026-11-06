from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录：
# 当前文件位于 app/core/config.py
# parents[2] 对应 RAG-learn 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    RAG 微服务的统一配置。

    配置优先级：
    系统环境变量 > .env 文件 > 默认值。
    """

    model_config = SettingsConfigDict(
        # 固定 .env 的位置，不依赖启动命令的工作目录
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",

        # 所有环境变量统一使用 RAG_ 前缀
        env_prefix="RAG_",

        # 环境变量名称不区分大小写
        case_sensitive=False,

        # 忽略空环境变量，避免空值覆盖默认值
        env_ignore_empty=True,

        # .env 中允许存在其他暂未使用的配置项
        extra="ignore",
    )

    # ==============================
    # 一、服务基础配置
    # ==============================

    app_name: str = "RAG Microservice"

    app_version: str = "0.1.0"

    app_env: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    api_v1_prefix: str = "/api/v1"

    # Dify 外部知识库接口（POST /retrieval）的鉴权密钥。
    # Dify 会在 Authorization 头里以 Bearer 形式发送它；两边填同一个值即可。
    # 这条路径不走 TokenAuthMiddleware（见 app/middleware/auth.py 的 EXCLUDED_PATHS），
    # 所以校验必须在路由内部做，不能指望中间件兜底。
    dify_retrieval_api_key: str = "dify-rag-key"

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # ==============================
    # 二、Embedding 配置（与 LLM 独立选择）
    # ==============================

    # 向量化实现，与 llm_provider 彼此独立：
    #   local -> 本地 Sentence Transformers（当前唯一已实现）
    # 将来要接云端 Embedding 时，在这里加一个取值，并在
    # app/services/embedding_client.py 的工厂里补对应分支。
    # 注意：取值刻意只列已实现的项，写了未实现的值会在启动时直接报错，
    # 而不是静默回落到本地模型。
    embedding_provider: Literal["local"] = "local"

    # 模型标识，仅用于日志与向量元数据（get_metadata），不参与加载路径。
    # 变量名与 .env 里的 RAG_EMBEDDING_MODEL 对应。
    embedding_model: str = "bge-m3"

    # 本地模型的存放路径，**必须是本机已存在的目录**。
    # 本服务只支持本地目录，不支持 HuggingFace 模型标识（如 BAAI/bge-m3）：
    # 那个写法会让 sentence-transformers 去联网下载，与「离线可用」的前提冲突。
    # 因此 create_embedding_backend() 在目录不存在时直接启动失败，
    # 而不是先启动成功、再由 /health/ready 一直报未就绪。
    # 便携式默认值，本机实际路径通过 .env 覆盖。
    embedding_model_path: Path = (
        PROJECT_ROOT / "models" / "bge-m3"
    )

    # 推理设备 cpu / cuda / mps。
    # 改造前该值硬编码在 EmbeddingConfig 里，配置层根本管不到。
    embedding_device: str = "cpu"

    # 向量化批大小
    embedding_batch_size: int = 32

    # ==============================
    # 三、LLM 配置（本地 Ollama / 云端 OpenAI 兼容服务）
    # ==============================

    # 生成模型实现：
    #   ollama            -> POST /api/chat，无鉴权
    #   openai_compatible -> POST {base_url}/chat/completions，Bearer 鉴权
    #                        （DeepSeek、通义千问兼容模式）
    # 选型只影响生成，不影响检索与向量库；而换 embedding 必须重建索引。
    llm_provider: Literal["ollama", "openai_compatible"] = "ollama"

    # 服务地址，取值随 provider 变化：
    #   ollama          -> http://127.0.0.1:11434
    #   DeepSeek        -> https://api.deepseek.com
    #   通义千问兼容模式 -> https://dashscope.aliyuncs.com/compatible-mode/v1
    # Docker 部署时通过环境变量覆盖为 host.docker.internal。
    llm_base_url: str = "http://127.0.0.1:11434"

    # 模型名称。默认值保持改造前的 qwen2.5:7b 不变。
    # 本机另装有 deepseek-r1:14b，如需切换请写进 .env，
    # 不要改这里的默认值，否则「重构前后行为一致」无法验证。
    # 云端可选 deepseek-chat / qwen-plus 等。
    llm_model: str = "qwen2.5:7b"

    # 云端服务的 API Key。provider=ollama 时留空。
    # 配了 openai_compatible 却没填会在启动时报错，而不是等第一次请求才 401。
    llm_api_key: Optional[str] = None

    # 模型请求超时时间，单位为秒
    llm_timeout_seconds: float = 180.0

    # 生成温度：制度问答要求可复现，取值宜低（0~0.3）
    # 注意：deepseek-reasoner 不接受该参数（传了会 400），切推理模型请用 deepseek-chat
    llm_temperature: float = 0.2

    # 单次回答的最大 token 数：防止长回答把单请求拖慢、拖垮排队中的其他请求
    # 对应 Ollama 的 num_predict、OpenAI 兼容的 max_tokens，语义一致
    # 400 token 约等于 300 个汉字，足够回答一条制度问题
    llm_max_tokens: int = 400

    # 聊天并发上限：超过后快速返回 503，避免请求无限排队
    # 单卡 GPU 实际串行处理，设太大会让人干等；建议 2~5
    llm_max_concurrency: int = 3

    # ==============================
    # 四、PostgreSQL 数据库配置
    # ==============================
    
    # PostgreSQL 连接字符串
    # 格式: postgresql+asyncpg://user:password@host:port/dbname
    # 默认使用本地开发环境的连接字符串
    # 生产环境通过环境变量覆盖(如 Docker Compose 中的 DATABASE_URL)
    database_url: str = "postgresql+asyncpg://rag_user:rag_secret_2024@localhost:5432/rag_db"

    # ==============================
    # 五、检索（RAG）策略
    # ==============================

    # 检索结果的最低相似度阈值，低于该值的片段不进入 prompt。
    #   score 语义：1 - 余弦距离，越大越相似，取值大致 0~1（重排最多 +0.05）。
    #   0 表示不过滤（默认值 = 保持改造前的行为，不引入回归）。
    # 调参依据（同一知识库、top_k=5 实测）：
    #   相关片段约 0.65~0.70；「那第二次呢」这类指代不清的问句 top1 只有 0.60
    #   且命中无关章节 —— 因此 0.5 附近开始有区分度，但不要一开始就设到 0.6 以上，
    #   会把「表述不同但内容正确」的片段一起丢掉。
    # 落地建议：先设 0 跑一段时间，从日志里看真实分数分布，再定阈值。
    retrieval_min_score: float = 0.0

    # ==============================
    # 六、认证与限流
    # ==============================

    # 认证实现。当前唯一取值 mock：测试阶段用「token 精确匹配表」确认调用方身份，
    # 不解析、不验签、不校验 Claims（见 app/core/mock_token_auth.py）。
    # 将来接入真实 Token 解析库时在这里加取值，并在 resources.py 补对应分支。
    auth_provider: Literal["mock"] = "mock"

    # 模拟 token 表，格式 "token1:user_001,token2:user_002"（逗号分隔，
    # 冒号前是 token、冒号后是 user_id）。仅 auth_provider=mock 时生效。
    # 留空会导致服务启动失败（见 resources.py 的前置校验）：
    # 认证配不齐时宁可起不来，也不要让一个「谁都没有身份」的服务静默对外。
    mock_tokens: str = ""

    # 说明：每人每分钟的请求限额已整体移除（见 app/middleware/auth.py 文件头）。
    # 原来这里有一个 rate_limit_per_minute 字段，现在不再读、不再生效。
    # 防刷只靠全局并发上限 llm_max_concurrency。

    # ==============================
    # 七、文档管理（上传 / 删除，仅管理员）
    # ==============================

    # 管理员名单：逗号分隔的 user_id，取自认证后的身份。
    # 留空 = 整个文档管理功能关闭（接口按「不存在」处理），服务仍正常启动。
    # 刻意让「空」等于「全关」而不是「全开」：配错最多功能不可用，
    # 不会变成所有人都能删库。
    admin_user_ids: str = ""

    # 允许上传的扩展名。只放解析器真正支持的两种：
    # .doc 是旧二进制格式，loaders 不支持，放开它只会让文件在解析阶段失败。
    upload_allowed_extensions: str = ".pdf,.docx"

    # 单文件大小上限（字节）。20 MB：制度类文档正常 < 5 MB，留足余量。
    upload_max_bytes: int = 20 * 1024 * 1024

    # 上传文件落地目录。服务端在这里为每次上传建 uuid 子目录，
    # 任务结束（成功或失败）立即删除，不保留原件。
    upload_temp_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # 任务记录保留时长（分钟）。任务表在进程内存里，进程重启即清空，
    # 但已入库的数据不受影响。
    ingest_task_ttl_minutes: int = 60

    # 任务表最多保留条数，超出后按创建时间淘汰最老的终态任务。
    ingest_task_max_records: int = 200

    @property
    def allowed_upload_extensions(self) -> frozenset[str]:
        """解析成小写扩展名集合，兼容 "pdf, .docx" 这种写法。"""
        items = (
            item.strip().lower() for item in self.upload_allowed_extensions.split(",")
        )
        return frozenset(
            item if item.startswith(".") else f".{item}" for item in items if item
        )

    @property
    def admin_user_id_set(self) -> frozenset[str]:
        """管理员 user_id 集合，自动去空白与空项。"""
        return frozenset(
            item.strip() for item in self.admin_user_ids.split(",") if item.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取进程级配置实例。

    正常运行时复用同一个配置对象。
    """
    return Settings()