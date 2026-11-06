"""对话接口的请求/响应模型。

只服务于「接收一轮对话」这一个动作。
会话与消息的模型在 app/schemas/conversation.py。
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.retrieval import RetrievalItem


class ChatRequest(BaseModel):
    """聊天请求。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户消息内容，最多 2000 字符",
    )
    session_id: Optional[str] = Field(
        None,
        pattern=r"^sess_[0-9a-f]{12}$",
        description=(
            "会话 ID。留空表示在「空对话窗口」发起首句："
            "服务端新建会话，返回的 session_id 就是这个窗口的归属，"
            "前端拿到后存入本地状态，后续消息带上它即可。"
        ),
    )


class ChatResponse(BaseModel):
    """聊天响应。"""

    session_id: str = Field(description="本轮对话归属的会话 ID")
    answer: str = Field(description="AI 生成的回答")
    sources: list[RetrievalItem] = Field(
        default_factory=list,
        description="本轮回答参考的知识片段",
    )
    message_count: Optional[int] = Field(
        None,
        description=(
            "该会话当前的消息总数（含本轮两条）。"
            "null 表示本轮保存失败，计数未知——此时不要再向数据库补查"
        ),
    )
    persisted: bool = Field(
        True,
        description=(
            "本轮对话是否已写入历史。用户消息与助手消息在同一事务内写入，"
            "因此只有「两条都在」和「两条都不在」两种状态，不存在半轮。"
            "False 表示答案可用但没有进历史，此时 message_count 为 null"
        ),
    )
    request_id: Optional[str] = Field(
        None,
        description=(
            "本次请求的追踪 ID，与响应头 X-Request-ID 一致；"
            "报障时提供它即可定位到对应日志"
        ),
    )
