"""会话资源接口：创建、标题历史、删除。

职责边界：
- 本文件只管「会话这个资源本身」：它什么时候产生、有哪些、什么时候消失
- 会话里说了什么（对话内容历史）在 messages.py
- 说一句话拿一个回答（对话动作）在 chat.py

删除统一用 POST，不提供 DELETE 方法：内部网关与前端代理对非 GET/POST 的
方法支持不一致，POST + 明确的动作路径最稳。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request

from app.api.deps import get_resources, require_user_id
from app.schemas.conversation import (
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionItem,
)
from src.session.session_manager import DeleteOutcome

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["会话管理"],
    responses={
        500: {"description": "服务器内部处理异常"},
        503: {"description": "服务尚未准备就绪"},
    },
)


# ── 创建 ──────────────────────────────────────────────────

@router.post(
    "",
    response_model=SessionItem,
    status_code=201,
    summary="创建会话",
    description=(
        "显式创建一个空会话并返回它的 ID，不触发检索和生成。\n\n"
        "注意：新建的空会话**不会**出现在会话标题历史里，"
        "因为它还没有任何消息；发出第一条消息后才会出现。"
        "若只是想在空窗口里开始对话，可以不调本接口，"
        "直接对 POST /chat 留空 session_id 由服务端建。"
    ),
    operation_id="create_conversation",
)
async def create_conversation(
    request: Request,
    payload: Optional[SessionCreateRequest] = None,
) -> SessionItem:
    """创建会话。会话归属当前 API Key 对应的用户。"""
    resources = get_resources(request)
    user_id = require_user_id(request)

    # create_session 用 INSERT ... RETURNING 直接返回插入行的字段，
    # 因此这里不需要再回读一次：回读一旦失败会误报 500，
    # 但会话其实已经提交，客户端重试就会多建一个孤儿会话
    info = await resources.session_manager.create_session(
        user_id=user_id,
        metadata=payload.metadata if payload else None,
    )

    logger.info("显式创建会话: %s user=%s", info["session_id"], user_id)
    # 刚创建的会话必然还没有消息，标题是空串
    return SessionItem(title="", **info)


# ── 会话标题历史 ──────────────────────────────────────────

@router.get(
    "",
    response_model=list[SessionItem],
    summary="会话标题历史",
    description=(
        "按最后活跃时间倒序返回当前用户的会话列表，每条带标题。\n\n"
        "标题取该会话首条用户消息的前 40 字，不是独立存储的字段。\n"
        "没有任何消息的空会话不会出现在这里。"
    ),
    operation_id="list_conversations",
)
async def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="最多返回多少条"),
    offset: int = Query(0, ge=0, description="跳过前多少条"),
) -> list[SessionItem]:
    """列出当前用户的会话（标题历史）。"""
    resources = get_resources(request)
    user_id = require_user_id(request)

    sessions = await resources.session_manager.list_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    return [SessionItem(**item) for item in sessions]


# ── 删除 ──────────────────────────────────────────────────

@router.post(
    "/{session_id}/delete",
    response_model=SessionDeleteResponse,
    summary="删除会话",
    description=(
        "删除指定会话。**逻辑删除**：只把会话标记为已删除，"
        "对话消息保留在数据库中，但所有接口都不再返回或使用它。\n\n"
        "删除后：会话列表不返回、消息历史 404、聊天接口拒绝继续使用该会话。\n\n"
        "会话不存在或不属于当前用户统一返回 404，"
        "不区分「没见过这个 ID」和「这是别人的会话」，避免被用来探测会话是否存在。\n\n"
        "**幂等**：重复删除同一个会话仍返回 200（不报错），"
        "这样客户端重试或用户连点不会看到「删除失败」。"
    ),
    operation_id="delete_conversation",
)
async def delete_conversation(
    request: Request,
    session_id: str = Path(
        ...,
        pattern=r"^sess_[0-9a-f]{12}$",
        description="会话 ID",
    ),
) -> SessionDeleteResponse:
    """逻辑删除会话（标记 deleted_at，不删消息）。"""
    resources = get_resources(request)
    user_id = require_user_id(request)

    # delete_session 内部已做归属校验：非归属者与不存在都返回 NOT_FOUND
    outcome = await resources.session_manager.delete_session(
        session_id,
        user_id=user_id,
    )

    if outcome is DeleteOutcome.NOT_FOUND:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    # 首次删除与重复删除都按成功返回：对调用方来说，
    # 「它现在是删除状态」这个事实是一致的，没必要让它区分第一次和第二次。
    if outcome is DeleteOutcome.ALREADY_DELETED:
        logger.info("重复删除已删除会话: %s user=%s", session_id, user_id)
    else:
        logger.info("逻辑删除会话: %s user=%s", session_id, user_id)

    return SessionDeleteResponse(session_id=session_id, deleted=True)


# ── 供 chat.py 复用的失败回收 ─────────────────────────────

async def discard_empty_session(resources, session_id: str, user_id: str) -> None:
    """回收「本轮请求自建、但没能写入任何消息」的空会话。

    只服务于 POST /chat 的失败路径：那时客户端还没拿到 session_id，
    删掉它对调用方不可见，可避免请求失败留下空壳会话堆积。

    显式创建的会话（POST /conversations）不走这里——那是用户主动要的，
    不该因为一次对话失败被悄悄删掉。

    这里刻意用 delete_session_if_empty（物理删除）而不是 delete_session
    （逻辑删除）：空会话没有任何用户可见内容，也没有需要保留的历史，
    物理删掉最干净。走逻辑删除只会留下「已删除且没有消息」的幽灵行，
    而这种行只有服务启动时的兜底清理才会扫到，等于要躺到下次重启。
    两个方法的分工就是「用户主动删的走逻辑删除，内部回收的走物理删除」。
    """
    try:
        removed = await resources.session_manager.delete_session_if_empty(
            session_id,
            user_id=user_id,
        )
        if removed:
            logger.info("已回收空会话: %s", session_id)
    except Exception:
        # 回收失败不应覆盖原始异常
        logger.warning("回收空会话失败: %s", session_id, exc_info=True)
