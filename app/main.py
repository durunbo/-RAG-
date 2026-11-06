import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routers.health import metrics_router, router as health_router
from app.api.routers.dify_retrieval import router as dify_retrieval_router
from app.api.routers.retrieval import router as retrieval_router
from app.api.routers.auth import router as auth_router
from app.api.routers.chat import router as chat_router
from app.api.routers.conversations import router as conversations_router
from app.api.routers.messages import router as messages_router
from app.api.routers.documents import router as documents_router
from app.core.config import get_settings
from app.core.resources import create_app_resources
from app.middleware.auth import TokenAuthMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.services.document_ingest import purge_stale_uploads

# ==========================================================
# 1. 加载统一配置
# ==========================================================

settings = get_settings()


# ==========================================================
# 2. 初始化日志
# ==========================================================

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# 3. 管理应用生命周期
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    管理 FastAPI 应用启动与关闭。

    后续在这里初始化和释放共享资源，例如：
    - Embedding 模型
    - PostgreSQL + pgvector 向量库
    - 数据库连接池
    - LLM 服务客户端

    当前阶段暂不加载这些资源。
    """

    logger.info(
        "服务启动：name=%s, version=%s, env=%s",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )

    try:
        # 一、创建共享资源对象
        resources = create_app_resources(settings)

        # 二、启动时真正加载 BGE-M3 模型
        # dimension 属性会触发 EmbeddingModel 内部的延迟加载
        embedding_dimension = resources.embedding_model.dimension

        # 三、验证 PostgreSQL 连接并获取向量总数
        await resources.vector_store.initialize()
        vector_count = await resources.vector_store.count()

        # 四、LLM 探活：只告警，不阻塞启动
        # 与 embedding 的 fail-fast 刻意区别对待：
        # - embedding 是本地模型目录，缺了就永远不可用，启动失败才说得清问题；
        # - LLM 是外部服务，可能比本服务晚就绪（Ollama 正在加载模型、云端实例
        #   刚扩出来）。此时硬失败会让容器反复重启（CrashLoopBackOff），
        #   比「起来了但 /health/ready 报未就绪」更难恢复。
        try:
            await asyncio.to_thread(resources.llm_client.ping)
            logger.info(
                "LLM 探活成功：provider=%s, model=%s",
                settings.llm_provider,
                settings.llm_model,
            )
        except Exception as exc:
            logger.warning(
                "LLM 探活失败：%s（服务仍启动，/health/ready 将返回 503）",
                exc,
            )

        # 五、所有关键资源检查通过后，再将它们交给应用使用
        app.state.resources = resources

        # 六、兜底清理：删除创建超过 1 小时仍没有任何消息的空会话
        # 清理失败不影响服务启动
        try:
            purged = await resources.session_manager.purge_empty_sessions(
                older_than_minutes=60
            )
            if purged:
                logger.info("已清理空会话：%s 条", purged)
        except Exception:
            logger.warning("空会话清理失败（不影响启动）", exc_info=True)

        # 七、清理上次进程残留的上传临时文件
        # 正常路径在任务 finally 里删，但 --reload、Ctrl+C、断电走不到那一步。
        # 不清理的话残留会慢慢占满磁盘，最后所有上传都失败
        try:
            removed = await asyncio.to_thread(
                purge_stale_uploads, resources.upload_temp_dir
            )
            if removed:
                logger.info("已清理残留上传文件：%s 项", removed)
        except Exception:
            logger.warning("残留上传文件清理失败（不影响启动）", exc_info=True)

        # 八、文档管理功能是否启用
        if resources.admin_user_ids:
            logger.info(
                "文档管理已启用：管理员 %s 位", len(resources.admin_user_ids)
            )
        else:
            logger.warning(
                "文档管理未启用：RAG_ADMIN_USER_IDS 为空，"
                "上传/查询/删除接口一律返回 404"
            )

        logger.info(
            "核心资源初始化完成：embedding_dimension=%s, vector_count=%s",
            embedding_dimension,
            vector_count,
        )

        yield

    finally:
        # 关闭数据库连接池（VectorStore 和 SessionManager 共享同一个 engine）
        if hasattr(app.state, "resources"):
            await app.state.resources.vector_store.close()
            await app.state.resources.db_engine.dispose()
            del app.state.resources

        logger.info("服务关闭：name=%s", settings.app_name)


# ==========================================================
# 4. 创建 FastAPI 应用
# ==========================================================

docs_enabled = settings.app_env != "production"

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="企业级 RAG 微服务 API",
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
    lifespan=lifespan,
)

# ==========================================================
# 5. 注册中间件
# ==========================================================

# 顺序有硬依赖：Starlette 里「后 add 的中间件在外层」。
# RequestContextMiddleware 必须在最外层，否则认证中间件直接返回 401 时，
# request.state 上还没有 request_id，错误响应里的追踪 ID 会是 null。
#
# 认证中间件只做「取 token → 验签 → 注入身份」，不再按分钟限流。
app.add_middleware(TokenAuthMiddleware)
app.add_middleware(RequestContextMiddleware)

# ── CORS：当前刻意不启用 ────────────────────────────────────
# 是否启用的唯一判断依据是「浏览器是否跨域直连本服务」：
#   - 同源反向代理（前端与 API 同一个域名端口）→ 浏览器认为同源，不需要 CORS
#   - 跨域直连（前端在别的域名/端口）        → 必须配白名单，否则请求全被浏览器拦掉
#   - 只有 curl / 脚本调用                  → 不涉及浏览器同源策略，配了也没意义
#
# 当前状态：尚无浏览器前端，只有命令行与脚本调用，因此不启用。
#
# 将来接入前端时，在这里打开（务必用显式白名单，不要用 "*"：
# 本项目用 Authorization 头传 Bearer token，不依赖 Cookie 自动携带，
# allow_credentials 保持 False）：
#
#     from fastapi.middleware.cors import CORSMiddleware
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["https://rag.example.com"],  # 前端实际来源，逐个列出
#         allow_methods=["GET", "POST"],
#         allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
#         allow_credentials=False,
#         max_age=600,
#     )
#
# 注意 CORS 是「浏览器同源策略」的放行规则，不是 API 的访问控制。
# 它挡不住 curl，真正的访问控制只有 token 验签。

# 统一错误出口：HTTPException / 参数校验失败 / 未捕获异常
register_exception_handlers(app)

# ==========================================================
# 6. 注册路由
# ==========================================================

# 健康检查不使用业务 API 版本前缀
app.include_router(health_router)
# Dify 外部知识库：路径固定为 /retrieval（Dify 会自动拼这个名字，不能加前缀）
app.include_router(dify_retrieval_router)
# 运行指标接口（需 Bearer token，不前缀 /health 以免被鉴权豁免规则放行）
app.include_router(metrics_router)

# 知识检索接口统一使用 /api/v1 前缀
app.include_router(
    retrieval_router,
    prefix=settings.api_v1_prefix,
)

# 认证接口：GET /auth/me，供前端显示当前登录身份
app.include_router(
    auth_router,
    prefix=settings.api_v1_prefix,
)

# 智能对话接口（同样使用 /api/v1 前缀）
# 只负责「接收一轮对话」；会话与消息是独立资源，分别挂下面两个
app.include_router(
    chat_router,
    prefix=settings.api_v1_prefix,
)

# 会话资源：创建 / 标题历史 / 删除
app.include_router(
    conversations_router,
    prefix=settings.api_v1_prefix,
)

# 消息资源：对话内容历史（只读）
app.include_router(
    messages_router,
    prefix=settings.api_v1_prefix,
)

# 文档管理（上传 / 查询进度 / 删除）：仅管理员，名单外的账号得到 404
app.include_router(
    documents_router,
    prefix=settings.api_v1_prefix,
)


# ==========================================================
# 7. 补充 OpenAPI 的认证声明
# ==========================================================
#
# 为什么需要单独补这一步：
#
# FastAPI 只会从「路由上的依赖」推断 securitySchemes。本项目的认证放在中间件里，
# 路由签名上没有 Security(...) 依赖，于是生成的规范里 securitySchemes 是空的
# （实测 `components.securitySchemes == {}`）。后果是 /docs 页面能打开、
# 但不出现 Authorize 按钮，无法填入 token，点任何「Try it out」都是 401 ——
# 文档页面成了摆设。
#
# 这段代码只影响 OpenAPI 文档本身，不参与任何请求校验。
# 真正的认证仍然只由 TokenAuthMiddleware 负责，改这里不会放宽或收紧权限。

_BEARER_SCHEME_NAME = "BearerAuth"

# 实际不需要 token 的路径：探针与文档自身（与中间件的 EXCLUDED_PATHS 对应）
_OPENAPI_EXEMPT_PATHS = ("/health", "/docs", "/redoc", "/openapi.json")


def _custom_openapi() -> dict:
    """生成 OpenAPI 规范，并声明 Bearer token 认证方式。"""
    # 已生成过就直接复用：FastAPI 默认行为相同，避免每个请求重复构建
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema.setdefault("components", {})["securitySchemes"] = {
        _BEARER_SCHEME_NAME: {
            "type": "http",
            "scheme": "bearer",
            "description": (
                "访问令牌。测试阶段为模拟 Token：与服务端 mock_tokens 配置"
                "精确匹配的字符串（见 .env.example）。"
                "Swagger 的 Authorize 里只需填令牌本身，不要自己加 \"Bearer \" 前缀"
                "（Swagger 会自动加）。"
            ),
        }
    }

    # 逐个接口声明 security。
    # 不使用全局 security，因为健康检查与文档页面确实不需要 token；
    # 若声明成「全局需要」，文档会与实际行为不符，反而误导调用方。
    for path, operations in schema.get("paths", {}).items():
        if path.startswith(_OPENAPI_EXEMPT_PATHS):
            continue

        for operation in operations.values():
            if isinstance(operation, dict):
                operation.setdefault(
                    "security",
                    [{_BEARER_SCHEME_NAME: []}],
                )

    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[method-assign]