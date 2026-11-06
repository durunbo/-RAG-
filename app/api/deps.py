"""路由层公共依赖。

这里放的是「每个路由都可能用到、却不属于任何业务域」的能力：

- get_resources  : 取应用生命周期里初始化的共享资源池
- require_user_id: 取认证中间件注入的用户身份
- track_metrics  : 请求级埋点（并发、耗时、结果）
- safe_request_id: 清洗外部传入的 X-Request-ID

集中在一处的目的不是少写几行，而是避免三个路由文件各写一份后行为漂移：
比如某天给身份校验加了「必须已绑定部门」，只改这里就全部生效。
"""

import functools
import logging
import time

from fastapi import HTTPException, Request

from app.core.metrics import metrics
from app.core.request_id import safe_request_id
from app.core.resources import AppResources

__all__ = [
    "get_resources",
    "require_user_id",
    "safe_request_id",
    "track_metrics",
]

logger = logging.getLogger(__name__)


def get_resources(request: Request) -> AppResources:
    """取应用共享资源。

    生命周期尚未完成初始化（例如 BGE-M3 还在加载）时返回 503，
    而不是让路由拿到 None 后抛 AttributeError 变成 500 纯文本。
    """
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="服务尚未准备就绪")
    return resources


def require_user_id(request: Request) -> str:
    """取认证中间件注入的 user_id。

    缺失说明中间件没生效或路径被错误豁免，属于服务端配置问题而不是调用方的错，
    所以返回 500 而不是 401。这里必须显式校验：直接取 request.state.user_id
    查不到属性时会抛 AttributeError，最终变成纯文本 500，前端 JSON.parse 会崩。
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        logger.error("API Key 认证中间件未注入 user_id")
        raise HTTPException(status_code=500, detail="认证信息缺失，请联系管理员")
    return user_id


# safe_request_id 的实现已下沉到 app.core.request_id：
# middleware 也需要它，而 middleware 不该反向依赖 app.api。
# 这里保留 re-export，既有调用方（路由层、脚本）不用改。


def track_metrics(func):
    """请求级埋点装饰器：统计并发、总耗时与成功失败。

    用装饰器而不是改函数体，是为了避免每次新增路由都重新调整缩进。
    被装饰的路由需要在 request.state.session_id 上写会话 ID（可选），
    本装饰器在日志里读取它用于关联。
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get("request")
        if request is None:
            request = next(
                (arg for arg in args if isinstance(arg, Request)),
                None,
            )

        # request_id 由 RequestContextMiddleware 统一分配，这里只读不生成：
        # 各处各自生成会让日志里的 id 与响应体、响应头的 id 对不上
        request_id = (
            getattr(request.state, "request_id", None) if request else None
        ) or safe_request_id(None)
        # enter() 返回的是进入时刻的并发深度；
        # leave() 之后 in_flight 已经减 1，那时再读就少算了一路
        depth_at_entry = metrics.enter()
        t_start = time.perf_counter()
        failed = True

        try:
            result = await func(*args, **kwargs)
            failed = False
            return result
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            metrics.leave(elapsed_ms, ok=not failed)
            logger.info(
                "chat_request request_id=%s session=%s elapsed_ms=%.1f "
                "status=%s in_flight_at_entry=%s",
                request_id,
                getattr(request.state, "session_id", None) if request else None,
                elapsed_ms,
                "ok" if not failed else "failed",
                depth_at_entry,
            )

    return wrapper
