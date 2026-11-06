"""从 Authorization: Bearer <token> 解析调用方身份。

身份来自调用方携带的 token，本服务不维护任何用户名录。
测试阶段使用模拟认证（token 精确匹配表），认证实现见 app/core/mock_token_auth.py；
为什么不用「服务端发 API Key」那套，见 scripts/init-db.sql 的说明段。

本文件只负责编排：
    取 token → 认证（app/core/mock_token_auth.py） → 注入身份 → 放行
认证细节、异常分类都在 mock_token_auth.py。

限流已于 2026-10-29 整体移除：前端是单人低频使用，20 次/分钟的额度
还不如页面一次刷新打的接口多（会话列表、消息列表、健康检查、身份校验），
结果是用户正常操作就被 429 打回，被当成「服务不稳定」。
真正的防刷仍然在：全局并发上限（RAG_LLM_MAX_CONCURRENCY）。

状态码约定（RFC 6750）：
    401 —— 没带 token、token 未知
    403 —— 认证通过但无权访问该资源（当前所有接口都只要求「是本人」，
           所以本中间件不产生 403）
    503 —— 认证未配置，属于部署问题，不是调用方的问题
"""

import logging

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.mock_token_auth import (
    TokenAuthNotConfigured,
    TokenError,
)
from app.core.error_response import build_error_response

logger = logging.getLogger(__name__)

# 401 响应统一带上这个头，是 RFC 6750 的要求：
# 客户端据此知道「该换一个 token 再来」，而不是把它当成普通的 400。
# 不写死 error 描述里可能含内部细节的部分，避免把校验失败原因回显给攻击者。
_WWW_AUTHENTICATE = 'Bearer realm="rag", error="invalid_token"'


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """解析 Bearer token，注入 user_id。

    完全豁免（不鉴权）：/health*、/docs、/redoc、/openapi.json
    """

    # /retrieval 是 Dify 外部知识库专用入口：它带的是独立的知识库密钥，
    # 不在 mock token 表里，所以必须绕开本中间件。
    # 该路径的鉴权在 app/api/routers/dify_retrieval.py 内部自己完成，
    # 不是「无鉴权」——只是换了另一套钥匙。
    EXCLUDED_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/retrieval"}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # ── 一、健康检查与文档豁免 ──────────────────────────────
        # 探针不带 token，且必须能在任何依赖故障时回答「我活着吗」，
        # 所以这一支在所有检查之前就返回
        path = request.url.path
        if path in self.EXCLUDED_PATHS or path.startswith("/health/"):
            return await call_next(request)

        # ── 二、确认应用资源已就绪 ──────────────────────────────
        # 生命周期尚未完成初始化时验不了 token，返回 503 而不是 500。
        # 用 getattr 而非属性访问：startup 失败时 app.state 上没有 resources
        resources = getattr(request.app.state, "resources", None)
        if resources is None:
            return build_error_response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                request=request,
                error_code="SERVICE_NOT_READY",
                message="服务尚未准备就绪，请稍后重试",
            )

        # ── 三、取出 Bearer token ───────────────────────────────
        token = self._extract_bearer_token(request.headers.get("Authorization"))

        if token is None:
            # 注意：此处必须直接返回响应，不能 raise HTTPException。
            # 本中间件位于 ExceptionMiddleware 外层，抛出的异常不会被转换成
            # JSON 响应，而是冒泡成 500。
            return build_error_response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                request=request,
                error_code="MISSING_TOKEN",
                message="缺少访问令牌，请在请求头添加 Authorization: Bearer <token>",
                headers={"WWW-Authenticate": _WWW_AUTHENTICATE},
            )

        request_id = getattr(request.state, "request_id", None)

        # ── 四、认证 ────────────────────────────────────────────
        try:
            principal = resources.token_auth.verify(token)
        except TokenAuthNotConfigured as exc:
            # 部署问题，不是调用方的问题。返回 503 让人去查配置，
            # 而不是 401 让所有人都去重新登录。
            logger.error(
                "token 认证未正确配置 request_id=%s path=%s error=%s",
                request_id,
                path,
                exc,
            )
            return build_error_response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                request=request,
                error_code="AUTH_NOT_CONFIGURED",
                message="认证服务未正确配置，请联系管理员",
            )
        except TokenError as exc:
            # 只记录失败类型，不记录 token 内容：
            # token 本身就是凭据，日志会被采集、会被更多人看到。
            # 排查靠 request_id 关联即可。
            logger.warning(
                "token 校验未通过 request_id=%s path=%s reason=%s",
                request_id,
                path,
                type(exc).__name__,
            )
            return build_error_response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                request=request,
                error_code="INVALID_TOKEN",
                message="访问令牌无效或已过期，请重新登录",
                headers={"WWW-Authenticate": _WWW_AUTHENTICATE},
            )

        # ── 五、认证器必须给出身份 ──────────────────────────────
        # 按契约，verify 要么返回 AuthPrincipal，要么抛 TokenError，
        # 不存在「返回 None 表示失败」这条路径。
        # 真出现了就是实现漂移：这里宁可 401 也不能带着 None 往下走，
        # 否则会在 request.state 赋值处变成 AttributeError（500），
        # 把「认证没给出结果」伪装成「服务内部错误」，掩盖真正原因。
        if principal is None:
            logger.error(
                "认证器返回了 None request_id=%s path=%s",
                request_id,
                path,
            )
            return build_error_response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                request=request,
                error_code="INVALID_TOKEN",
                message="访问令牌无效或已过期，请重新登录",
                headers={"WWW-Authenticate": _WWW_AUTHENTICATE},
            )

        # ── 六、注入身份 ────────────────────────────────────────
        # user_id 保持扁平字段：路由层 require_user_id 只认它，
        # 换认证方案不应该波及所有路由的取值方式
        request.state.user_id = principal.user_id
        # principal 额外带上展示名与部门，供 /auth/me 使用
        request.state.principal = principal

        return await call_next(request)

    @staticmethod
    def _extract_bearer_token(authorization: str | None) -> str | None:
        """从 Authorization 头取出 Bearer token，取不到返回 None。

        对 scheme 大小写不敏感是刻意的：RFC 7235 规定认证方案名不区分大小写，
        `bearer`、`Bearer`、`BEARER` 都应接受。有些客户端库发的就是小写，
        如果这里严格匹配，对接时会变成一个很难查的「偶发 401」。
        """
        if not authorization:
            return None

        parts = authorization.split(None, 1)
        if len(parts) != 2:
            return None

        scheme, credentials = parts
        if scheme.lower() != "bearer":
            return None

        token = credentials.strip()
        return token or None
