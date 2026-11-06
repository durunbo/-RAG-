"""消息资源接口：对话内容历史（只读）。

职责边界：只负责「把这个会话说过的话按时间顺序读出来」。
不提供修改，也不提供清空——历史是既成事实，要清空就删整个会话
（POST /conversations/{session_id}/delete）。
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Query, Request

from app.api.deps import get_resources, require_user_id
from app.schemas.conversation import MessageItem

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["消息历史"],
    responses={
        500: {"description": "服务器内部处理异常"},
        503: {"description": "服务尚未准备就绪"},
    },
)


@router.get(
    "/{session_id}/messages",
    response_model=list[MessageItem],
    summary="对话内容历史",
    description=(
        "按时间正序返回指定会话的消息，包含用户提问、AI 回答，"
        "以及每条回答当时引用的检索片段快照。\n\n"
        "`limit` 取的是**最近** N 条而不是最早 N 条，"
        "所以长会话下返回的是最新的一段，顺序仍然是旧→新。\n\n"
        "会话不存在或不属于当前用户统一返回 404。"
    ),
    operation_id="list_messages",
)
async def list_messages(
    request: Request,
    session_id: str = Path(
        ...,
        pattern=r"^sess_[0-9a-f]{12}$",
        description="会话 ID",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="返回最近多少条消息",
    ),
) -> list[MessageItem]:
    """读取指定会话的对话内容历史。"""
    resources = get_resources(request)
    user_id = require_user_id(request)

    # 先校验归属：不存在与无权限统一 404，不泄露会话存在性
    if not await resources.session_manager.session_exists(
        session_id,
        user_id=user_id,
    ):
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    messages = await resources.session_manager.get_history(
        session_id=session_id,
        user_id=user_id,
        limit=limit,
    )
    return [MessageItem(**item) for item in messages]
