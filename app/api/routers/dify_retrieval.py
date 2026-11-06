"""Dify 外部知识库适配接口（POST /retrieval）。

Dify 的「连接外部知识库」只能接一种固定协议，字段名和本项目的
/api/v1/retrieval/search 完全不同：

    请求  {"knowledge_id": "...", "query": "...",
           "retrieval_setting": {"top_k": 4, "score_threshold": 0.5}}
    响应  {"records": [{"content": "...", "score": 0.87,
                        "title": "...", "metadata": {}}]}

协议说明见 Dify 文档《连接外部知识库》。

为什么不直接改 /api/v1/retrieval/search：那个接口的 results/text/source
是已经固化的前端契约，改字段名会波及前端。这里只做一层格式翻译，
检索仍然调用同一个 vector_store.search，所以两套接口看到的结果一致。

两个 Dify 侧容易踩的点，在这里兜住：
1. records[].metadata 必须是对象，不能是 null —— 传 null 会让它的
   检索链路报错，而错误信息不会告诉你原因。
2. 它保存 API 配置时会先发一次测试请求，那个请求的 query 可能为空。
   这里对空 query 返回空 records 而不是 400，避免「测试连接」这一步失败。
"""

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dify 外部知识库"])

# Dify 规范里的错误码。HTTP 状态码同时给出，便于用 curl 排查。
_ERR_AUTH = 1001
_ERR_PARAM = 1002
_ERR_NOT_FOUND = 2001

# 保护上限：Dify 的 top_k 是用户在界面上拖的滑块，可能被拖到很大。
_MAX_TOP_K = 20


class _RetrievalSetting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_k: int = Field(default=4, ge=1)
    score_threshold: float | None = Field(default=None)


class _DifyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # knowledge_id 由 Dify 侧「外部知识库 ID」输入框决定。
    # 本项目只有一个库，这里只用于日志排查，不参与检索。
    knowledge_id: str = Field(default="")
    query: str = Field(default="")
    retrieval_setting: _RetrievalSetting = Field(default_factory=_RetrievalSetting)


def _error(status: int, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error_code": code, "error_msg": message},
    )


def _check_auth(request: Request, expected: str) -> bool:
    """校验 Bearer 密钥。

    不用 app.core.mock_token_auth：那是给前端用的用户 token 表，
    这里要的是一个独立的知识库密钥，混用会让「Dify 能读哪些库」这件事
    变得不受控。
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return False
    return header[7:].strip() == expected


@router.post(
    "/retrieval",
    summary="Dify 外部知识库检索",
    description=(
        "供 Dify「连接外部知识库」调用的检索接口。"
        "请求/响应格式遵循 Dify External Knowledge API 协议。"
    ),
    operation_id="dify_external_retrieval",
)
async def dify_external_retrieval(request: Request) -> Any:
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        return _error(503, _ERR_NOT_FOUND, "检索服务尚未就绪")

    # 密钥从全局设置读，不走 resources：AppResources 不持有 config
    # （embedding / llm / 上传限制都是各自字段），硬塞进去会多一条无用的传递链。
    from app.core.config import get_settings

    expected_key = get_settings().dify_retrieval_api_key
    if not _check_auth(request, expected_key):
        # 不回显正确密钥，也不区分「没带」和「带错」
        return _error(401, _ERR_AUTH, "API Key 无效")

    try:
        raw_body = await request.json()
    except Exception:  # noqa: BLE001
        return _error(400, _ERR_PARAM, "请求体不是合法 JSON")

    if not isinstance(raw_body, dict):
        return _error(400, _ERR_PARAM, "请求体必须是对象")

    try:
        payload = _DifyRequest.model_validate(raw_body)
    except Exception as exc:  # noqa: BLE001
        return _error(400, _ERR_PARAM, f"参数校验失败：{exc}")

    query = payload.query.strip()
    setting = payload.retrieval_setting
    top_k = max(1, min(setting.top_k, _MAX_TOP_K))

    # Dify 保存 API 配置时的连通性测试可能不带 query。
    # 返回空结果让它通过，而不是用 400 把「测试连接」这一步挡死。
    if not query:
        logger.info("Dify 检索收到空 query，返回空结果（通常是连通性测试）")
        return {"records": []}

    try:
        raw_results = await resources.vector_store.search(query=query, top_k=top_k)
    except Exception:
        logger.exception("Dify 外部检索执行失败")
        return _error(503, _ERR_NOT_FOUND, "检索暂时不可用")

    threshold = setting.score_threshold
    records = []
    for item in raw_results:
        try:
            score = float(item["score"])
            text = item["text"]
        except (KeyError, TypeError, ValueError):
            logger.warning("跳过格式异常的检索结果：%r", item)
            continue

        # threshold 为 None 或 0 时不过滤。Dify 界面上默认 0.5，
        # 而本库正样本最低分约 0.56、负样本最高约 0.69，两者重叠，
        # 开阈值会把一部分正确答案切掉，建议在 Dify 侧关闭。
        if threshold is not None and threshold > 0 and score < threshold:
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        source_file = str(metadata.get("source_file") or "")
        section = str(metadata.get("section") or "")
        title = f"{source_file} / {section}" if section else source_file

        records.append(
            {
                "content": text,
                "score": score,
                "title": title or "企业制度",
                # 必须是非 null 对象：Dify 收到 null 会在检索链路里报错
                "metadata": {
                    "source_file": source_file,
                    "document_type": str(metadata.get("document_type") or ""),
                    "heading_path": str(metadata.get("heading_path") or ""),
                    "section": section,
                    "chunk_index": metadata.get("chunk_index"),
                    "question": str(metadata.get("question") or ""),
                },
            }
        )

    logger.info(
        "Dify 外部检索：query=%r knowledge_id=%s top_k=%s 命中=%s",
        query,
        payload.knowledge_id,
        top_k,
        len(records),
    )
    return {"records": records}
