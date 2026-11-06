"""认证接口。

只有一件事：告诉调用方「你手上这个 token 对应谁」。

不提供登录、刷新、发令牌——这些都由公司既有登录系统负责。
本服务只验签、取身份，不签发任何凭据。
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import get_resources, require_user_id
from app.schemas.auth import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["认证"],
    responses={
        500: {"description": "服务器内部处理异常"},
        503: {"description": "服务尚未准备就绪"},
    },
)


@router.get(
    "/me",
    response_model=CurrentUser,
    summary="当前身份",
    description=(
        "返回本次请求所用 token 对应的用户身份。\n\n"
        "前端可以用它显示「当前登录人」。\n"
        "这个接口本身也需要 `Authorization: Bearer <token>`，"
        "但不计入每分钟的请求额度。\n\n"
        "`display_name` / `department` 只有在 token 里包含对应字段时才有值，"
        "否则返回 null。"
    ),
    operation_id="get_current_user",
)
async def get_current_user(request: Request) -> CurrentUser:
    """读取中间件已解析出的身份。"""
    resources = get_resources(request)
    user_id = require_user_id(request)

    principal = getattr(request.state, "principal", None)
    if principal is None:
        # 中间件只注入了 user_id 却没注入 principal，属于实现漂移。
        # 这里不拿 user_id 兜底拼一个假身份：展示信息都没有，
        # 拼出来的对象会让前端以为「这人没有部门」，掩盖真实问题。
        logger.error("认证中间件未注入 principal user=%s", user_id)
        raise HTTPException(status_code=500, detail="认证信息不完整")

    return CurrentUser(
        user_id=principal.user_id,
        display_name=principal.display_name,
        department=principal.department,
        # 限流已移除：字段保留是为了不破坏已有调用方的响应结构，
        # 恒定返回 0（0 = 未启用限流）。前端不再展示这个值。
        rate_limit_per_minute=0,
        is_admin=resources.is_admin(principal.user_id),
    )
