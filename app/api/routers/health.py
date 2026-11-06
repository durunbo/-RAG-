import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

from app.core.config import get_settings
from app.core.metrics import metrics
from app.schemas.health import HealthResponse


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["健康检查"],
)

# 运行指标使用独立 router：不挂在 /health 下，
# 避免被认证中间件的「/health/* 放行」规则跳过鉴权
metrics_router = APIRouter(tags=["监控"])


@metrics_router.get("/metrics", summary="运行指标", operation_id="get_metrics")
def get_metrics() -> dict:
    """返回进程内运行指标（并发峰值、延迟、失败数）。"""
    return metrics.snapshot()


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="服务存活检查",
)
def check_liveness() -> HealthResponse:
    """
    存活检查：确认应用进程能够响应请求。

    不检查模型、数据库等业务依赖，
    避免依赖故障时误触发应用重启。
    """
    settings = get_settings()

    return HealthResponse(
        status="alive",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="服务就绪检查",
)
async def check_readiness(request: Request) -> HealthResponse:
    """
    就绪检查：确认向量化模型、向量库、数据库与大模型服务都可用于处理请求。

    与 /health/live 的分工：
    - live  只证明进程活着，依赖故障不重启应用；
    - ready 真实探活全部依赖，任一不可用返回 503，由负载均衡摘流。

    不通过时用不同的 detail 区分原因（模型目录 / 核心资源 / 数据库 / LLM），
    因为这几种故障的处置动作完全不同。

    注意：不检查认证。认证是验签调用方的 token，不依赖数据库、
    也不需要任何本地表——token 方案下「库里没有用户表」是正常的。
    """
    settings = get_settings()

    # 一、确认应用资源已初始化
    resources = getattr(request.app.state, "resources", None)

    if resources is None:
        raise HTTPException(
            status_code=503,
            detail="应用核心资源尚未初始化",
        )

    # 二、确认模型目录仍然存在
    # 启动时已由 create_embedding_backend() 强校验过一次（不存在直接启动失败），
    # 这里防的是运行期挂载丢失：Docker volume 所在磁盘被卸载、网络盘断连等。
    if not settings.embedding_model_path.is_dir():
        raise HTTPException(
            status_code=503,
            detail="Embedding 模型目录不可用",
        )

    # 三、检查模型维度和向量库读操作
    try:
        dimension = resources.embedding_model.dimension
        await resources.vector_store.count()

        if dimension <= 0:
            raise RuntimeError("Embedding 模型维度无效")

    except Exception:
        logger.exception("服务就绪检查失败")

        raise HTTPException(
            status_code=503,
            detail="核心资源检查失败，服务暂不可用",
        ) from None

    # 四、确认数据库可用
    #
    # 只验证「连得上」，不检查任何认证相关的表：
    # 用户身份由调用方 token 提供，本服务不再维护用户名录，
    # 数据库只负责会话与消息的持久化。
    try:
        async with resources.db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        logger.exception("数据库就绪检查失败")

        raise HTTPException(
            status_code=503,
            detail="数据库不可用",
        ) from None

    # 五、确认 LLM 真的可用（不是只看配置）
    # 同步客户端交给线程池，避免探活阻塞事件循环。
    # 不与上面的 embedding 检查合并：两者失败要给出不同的原因，
    # 「模型没加载」和「大模型服务连不上」的处置动作完全不同。
    try:
        await asyncio.to_thread(resources.llm_client.ping)
    except Exception as exc:
        # 不打整段堆栈：探针会周期性地打这一行，堆栈会把日志淹掉。
        # 具体原因（鉴权失败 / 模型未拉取 / 连不上）已经在异常消息里。
        logger.warning("LLM 就绪检查失败：%s", exc)

        raise HTTPException(
            status_code=503,
            detail="LLM 服务不可用",
        ) from None

    # 六、所有检查通过后，才返回 ready
    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
    )