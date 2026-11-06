import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.schemas.retrieval import (
    RetrievalItem,
    RetrievalRequest,
    RetrievalResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/retrieval",
    tags=["知识检索"],
    responses={
        500: {"description": "服务器内部处理异常"},
        503: {"description": "检索服务暂不可用"},
    },
)


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="检索企业知识库",
    description=(
        "根据用户问题，从 PostgreSQL + pgvector 知识库检索相关知识片段。"
        "本接口只负责检索，不负责生成最终答案。"
    ),
    operation_id="search_knowledge",
)
async def search_knowledge(
    payload: RetrievalRequest,
    request: Request,
) -> RetrievalResponse:
    """
    知识检索接口。

    处理流程：
    1. 获取 FastAPI 生命周期中初始化的共享资源。
    2. 调用现有 VectorStore.search() 执行检索。
    3. 校验并转换检索结果。
    4. 返回统一的响应结构。
    """

    # 一、获取应用共享资源
    resources = getattr(
        request.app.state,
        "resources",
        None,
    )

    if resources is None:
        raise HTTPException(
            status_code=503,
            detail="检索服务尚未准备就绪",
        )

    # 二、执行向量检索
    try:
        raw_results = await resources.vector_store.search(
            query=payload.query,
            top_k=payload.top_k,
        )

    except Exception:
        # 详细异常只记录在服务端日志中
        # 不将内部路径、数据库错误等信息返回给调用方
        logger.exception("知识检索执行失败")

        raise HTTPException(
            status_code=503,
            detail="知识检索暂时不可用",
        ) from None

    # 三、校验并转换检索结果
    # 字段清洗（空串 → None）在 RetrievalItem.from_raw 里统一做，
    # 这里不再手写一遍——正是那份手写逻辑漏了清洗才导致接口 500
    try:
        formatted_results = [
            RetrievalItem.from_raw(result) for result in raw_results
        ]

    except (KeyError, TypeError, ValueError, ValidationError):
        logger.exception("知识检索结果格式校验失败")

        raise HTTPException(
            status_code=500,
            detail="检索结果格式异常",
        ) from None

    # 四、构造统一响应
    return RetrievalResponse(
        query=payload.query,
        requested_top_k=payload.top_k,
        returned_count=len(formatted_results),
        results=formatted_results,
    )