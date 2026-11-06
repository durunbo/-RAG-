"""统一错误响应的构造。

所有错误响应都是同一份形状（四个字段恒定存在）：
    {"error_code": "...", "message": "...", "request_id": "...", "details": null|[...]}

放在 app/core 而不是 app/api，是因为认证中间件也要用它：
middleware 一旦 import app.api.*，就会把整个 API 层拖进中间件，
而 API 层将来又会反过来依赖 middleware，形成分层倒置与潜在循环导入。
依赖方向保持 core ← api、core ← middleware，两边都只向下看。
"""

import logging
from typing import Mapping, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


# 状态码 → 默认错误码。只列本项目真会出现的状态码，其余回落 ERROR，
# 避免维护一张永远对不上的大表。
STATUS_CODE_TO_ERROR_CODE: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def build_error_response(
    *,
    status_code: int,
    request: Optional[Request] = None,
    error_code: Optional[str] = None,
    message: Optional[str] = None,
    headers: Optional[Mapping[str, str]] = None,
    details: Optional[list[dict]] = None,
) -> JSONResponse:
    """构造统一错误响应。中间件与异常处理器共用这一个函数。"""
    request_id = (
        getattr(request.state, "request_id", None)
        if request is not None
        else None
    )

    body = ErrorResponse(
        error_code=error_code or STATUS_CODE_TO_ERROR_CODE.get(status_code, "ERROR"),
        message=message or "请求处理失败",
        request_id=request_id,
        details=details,
    )

    # 响应头在这里自己写，不依赖 RequestContextMiddleware 的 send 包装：
    # 未捕获异常产生的 500 由 ServerErrorMiddleware 负责生成响应，
    # 它位于最外层、不经过业务中间件，否则会出现
    # 「响应体里有 request_id、响应头里没有」的割裂情况。
    # setdefault 保证调用方传入的 Retry-After 等头部原样保留、且不重复写。
    response_headers = dict(headers or {})
    if request_id:
        response_headers.setdefault("X-Request-ID", request_id)

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers=response_headers or None,
    )
