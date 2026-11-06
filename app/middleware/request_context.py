"""请求上下文中间件：给每个请求分配 request_id 并在响应头回显。

为什么不塞进认证中间件：
- 认证中间件会跳过 /health、/docs，这些路径同样需要 request_id
- 认证是「能不能进」，request_id 是「怎么找到这次请求」，两件事

为什么用纯 ASGI 而不是 BaseHTTPMiddleware：
BaseHTTPMiddleware 会为每个请求额外起一个 task 并包装 send，
将来做 SSE 流式输出时它会干扰数据流的结束时机。
这里只做两件事——在 scope 里塞 id、在响应头回显，不碰数据流。
"""

from starlette.datastructures import MutableHeaders
from starlette.requests import Request

from app.core.request_id import safe_request_id


class RequestContextMiddleware:
    """把 request_id 写进 scope["state"]，并在响应头回显。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = safe_request_id(
            Request(scope).headers.get("X-Request-ID")
        )

        # scope["state"] 就是 request.state 的底层字典：
        # 写在这里之后，后续中间件、路由、异常处理器读到的是同一个 id
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                # 用赋值而不是 append，避免和 build_error_response 里
                # 已经写好的 X-Request-ID 重复
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)
