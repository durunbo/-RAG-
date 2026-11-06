"""对话接口：只负责「接收一轮对话并返回回答」。

这个文件里只有一件事：给我一句话，我给你答案。
会话的创建 / 列表 / 删除在 conversations.py；
对话内容历史在 messages.py。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import get_resources, require_user_id, track_metrics
from app.api.routers.conversations import discard_empty_session
from app.core.metrics import metrics
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievalItem
from app.services.rag_service import (
    GenerationFailed,
    RetrievalUnavailable,
    ServiceBusy,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["智能对话"],
    responses={
        500: {"description": "服务器内部处理异常"},
        503: {"description": "服务暂不可用"},
    },
)


# ── 辅助函数 ──────────────────────────────────────────────

def _format_sources(raw_sources: list[dict]) -> list[RetrievalItem]:
    """把服务层返回的原始字典转成 API schema。

    字段清洗（空字符串 → None）统一在 RetrievalItem.from_raw 里做，
    这里只负责批量包装。以前两个路由各写一份转换逻辑，
    retrieval 那份漏了清洗，直接导致 /retrieval/search 500。
    """
    return [RetrievalItem.from_raw(item) for item in raw_sources]


def _http_error_from(exc: Exception) -> HTTPException:
    """把服务层异常翻译成 HTTP 响应。

    映射集中在这里，新增异常类型时只需要改一处；
    末尾的兜底分支保证将来漏配的异常不会静默变成空响应。
    """
    if isinstance(exc, ServiceBusy):
        metrics.record_busy()
        return HTTPException(
            status_code=503,
            detail="服务繁忙，请稍后重试",
            headers={"Retry-After": "5"},
        )
    if isinstance(exc, RetrievalUnavailable):
        return HTTPException(status_code=503, detail="知识检索暂时不可用")
    if isinstance(exc, GenerationFailed):
        return HTTPException(status_code=500, detail="AI 回答生成失败")
    logger.error("未映射的服务层异常: %s", type(exc).__name__)
    return HTTPException(status_code=500, detail="服务内部错误")


async def _run_turn(
    *,
    resources,
    request: Request,
    user_id: str,
    session_id: str,
    message: str,
    top_k: int,
    is_new_session: bool,
) -> ChatResponse:
    """执行一轮问答：读历史 → 检索 → 生成 → 落库 → 组装响应。

    Args:
        is_new_session: True 表示这个会话是本次请求刚建的。
            为 True 时本轮失败会回收空会话；显式创建的会话传 False，
            不能因为一次对话失败就把用户的会话删掉。
    """
    # 供埋点装饰器输出日志时关联
    request.state.session_id = session_id
    # 一、历史消息（滑动上下文窗口）
    history = await resources.session_manager.get_history(
        session_id=session_id,
        user_id=user_id,
        limit=resources.rag_service.config.context_window,
    )

    # 二、检索 + 生成
    try:
        rag_result = await resources.rag_service.answer(
            query=message,
            history=history,
            top_k=top_k,
        )
    except (ServiceBusy, RetrievalUnavailable, GenerationFailed) as exc:
        if is_new_session:
            await discard_empty_session(resources, session_id, user_id)
        raise _http_error_from(exc) from None

    metrics.record_stage(rag_result.retrieval_ms, rag_result.generation_ms)

    # 三、来源格式化
    formatted_sources = _format_sources(rag_result.sources)

    # 四、落库：整轮原子写入
    # 保存失败不吞掉答案：答案已经生成，重试一次成本很高，
    # 因此返回 200 但在响应里标 persisted=False，让调用方知道这一轮查不到。
    #
    # 这里刻意保持「写失败后不再读数据库」：数据库正处在故障中时，
    # 任何后续查询都会二次失败，把已经生成好的答案一起丢掉。
    # 所以 message_count 直接留 None（表示未知），不再调 get_message_count。
    persisted = True
    message_count: Optional[int] = None
    try:
        _, _, message_count = await resources.session_manager.add_turn(
            session_id=session_id,
            user_id=user_id,
            user_content=message,
            assistant_content=rag_result.answer,
            retrieved_chunks=[item.model_dump() for item in formatted_sources],
        )
    except ValueError:
        # 生成期间会话被删除/被转走：属于并发下可预期的情况，
        # 不是服务故障，用 warning 记一句就够，不必打整段堆栈
        persisted = False
        logger.warning("会话在生成期间消失，本轮未入库 session=%s", session_id)
    except Exception:
        persisted = False
        logger.exception("保存对话失败 session=%s", session_id)

    return ChatResponse(
        session_id=session_id,
        answer=rag_result.answer,
        sources=formatted_sources,
        message_count=message_count,
        persisted=persisted,
        # 与响应头 X-Request-ID 同源（都由 RequestContextMiddleware 分配），
        # 前端报障时可以直接把这个字段贴过来
        request_id=getattr(request.state, "request_id", None),
    )


# ── 对话接口 ──────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatResponse,
    summary="发送消息并获取回答",
    description=(
        "接收一轮对话：检索知识库、调用 LLM 生成回答，并把这一轮写入历史。\n\n"
        "**会话由谁创建**：`session_id` 留空时，服务端为这次请求新建会话并在响应里"
        "返回新 ID——这就是「空对话窗口里直接开聊」的用法，前端拿到 ID 存入本地状态，"
        "后续消息带上它即可。\n\n"
        "`session_id` 非空时只校验归属，**不会**因为查不到就偷偷新建。"
    ),
    operation_id="send_message",
)
@track_metrics
async def send_message(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """发送消息并获取 AI 回答。"""
    resources = get_resources(request)
    user_id = require_user_id(request)

    session_id = payload.session_id
    is_new_session = False

    if not session_id:
        created = await resources.session_manager.create_session(user_id=user_id)
        session_id = created["session_id"]
        is_new_session = True
        logger.info("对话接口新建会话: %s user=%s", session_id, user_id)
    elif not await resources.session_manager.session_exists(
        session_id,
        user_id=user_id,
    ):
        # 归属不符与不存在同样处理，不泄露会话存在性
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    # 检索条数不再由调用方指定：RAGService 最终只把前 max_sources 段拼进 prompt，
    # 传更小的值只会少给资料，传更大的值会白跑一次大范围向量检索。
    top_k = resources.rag_service.config.max_sources

    return await _run_turn(
        resources=resources,
        request=request,
        user_id=user_id,
        session_id=session_id,
        message=payload.message,
        top_k=top_k,
        is_new_session=is_new_session,
    )
