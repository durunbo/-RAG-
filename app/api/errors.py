"""注册全局异常处理器。

响应构造本身在 app/core/error_response.py（认证中间件也要用，不能放 API 层）。
这里只负责把四个出口挂到 FastAPI 上，缺一个就会有漏网的错误格式：

1. HTTPException          —— 业务代码主动抛（404 会话不存在、503 服务繁忙…）
2. RequestValidationError —— 请求体/查询参数不合法（422）
3. Exception              —— 兜底（500）

前两个由内层 ExceptionMiddleware 处理，响应会经过业务中间件；
第三个由最外层 ServerErrorMiddleware 处理，不经过业务中间件——
所以 X-Request-ID 响应头必须在 build_error_response 里写，不能只靠中间件。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.core.error_response import build_error_response

logger = logging.getLogger(__name__)


def _first_error_location(errors: list[dict]) -> str:
    """把 pydantic 的 loc 元组拼成 "message" / "metadata.size" 这种路径。

    loc 形如 ("body", "message")，"body" 这段对调用方没有意义，去掉。
    """
    first = errors[0] if errors else {}
    return ".".join(
        str(part) for part in first.get("loc", ()) if part != "body"
    )


async def _handle_http_exception(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """业务异常：detail 通常是字符串，也允许是结构化 dict。"""
    detail = exc.detail
    if isinstance(detail, dict):
        error_code = detail.get("error_code")
        message = detail.get("message")
    else:
        error_code = None
        message = str(detail)

    return build_error_response(
        status_code=exc.status_code,
        request=request,
        error_code=error_code,
        message=message,
        # 必须透传：503 的 Retry-After 丢了，客户端就不会退避重试
        headers=getattr(exc, "headers", None),
    )


async def _handle_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """参数校验失败：保留字段级明细，但只回显定位路径与原因。

    只放 loc / msg / type 三个字段，不回显 input 和 ctx——
    那里面是用户提交的原文（可能是超长文本、手机号等），
    写进错误响应等于把敏感输入再吐回去一次。
    """
    errors = exc.errors()
    loc = _first_error_location(errors)
    reason = errors[0].get("msg", "请求参数不合法") if errors else "请求参数不合法"

    details = [
        {
            "loc": ".".join(
                str(part) for part in err.get("loc", ()) if part != "body"
            ),
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in errors
    ]

    return build_error_response(
        status_code=422,
        request=request,
        error_code="VALIDATION_ERROR",
        message=f"{loc}: {reason}" if loc else reason,
        details=details,
    )


async def _handle_unhandled_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """兜底：只记日志，不回显异常内容。

    堆栈里可能带文件路径、SQL、连接串，直接返回给调用方等于泄露内部结构。
    """
    logger.exception(
        "未捕获异常 request_id=%s method=%s path=%s",
        getattr(request.state, "request_id", None),
        request.method,
        request.url.path,
    )
    return build_error_response(
        status_code=500,
        request=request,
        error_code="INTERNAL_ERROR",
        message="服务器内部错误，请稍后重试",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """把三个出口挂到应用上，在 main.py 建完 app 后调用一次。"""
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    # FastAPI 会把它交给最外层的 ServerErrorMiddleware
    app.add_exception_handler(Exception, _handle_unhandled_exception)
