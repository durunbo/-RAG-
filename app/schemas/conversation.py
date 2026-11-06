"""会话与消息的对外数据模型。

重要：本文件只定义「响应 JSON 长什么样」，不存储任何数据。
会话和消息的真实数据保存在 PostgreSQL 的 sessions / messages 表里，
这里的类每次请求把数据库查出来的行「塑形」成 JSON，用完即弃。

命名上没有用 History 一类的字眼，是因为这些类同时也用于「刚创建、还没有任何
消息」的会话，叫 History 会误导调用方以为它跟历史记录有关。
"""

import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── 请求模型 ──────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    """创建会话的请求体，可整体省略（等价于不传 metadata）。"""

    model_config = ConfigDict(extra="forbid")

    metadata: dict = Field(
        default_factory=dict,
        description="会话扩展元数据，原样写入 sessions.metadata_json",
    )

    @field_validator("metadata")
    @classmethod
    def _limit_metadata(cls, value: dict) -> dict:
        """限制体积，避免客户端塞进超大 JSON 把这一行撑爆。"""
        if len(json.dumps(value, ensure_ascii=False)) > 2000:
            raise ValueError("metadata 序列化后不得超过 2000 字符")
        return value


# ── 响应模型 ──────────────────────────────────────────────

class SessionItem(BaseModel):
    """一条会话的对外表示（用于创建会话的响应与会话标题历史）。"""

    session_id: str = Field(description="会话 ID")
    user_id: str = Field(description="归属用户")
    title: str = Field(
        "",
        description="标题：取首条用户消息前缀；空串表示这个会话还没有任何消息",
    )
    created_at: Optional[str] = Field(None, description="创建时间（ISO 8601）")
    updated_at: Optional[str] = Field(None, description="最后活跃时间（ISO 8601）")
    metadata: dict = Field(default_factory=dict, description="扩展元数据")


class SessionDeleteResponse(BaseModel):
    """删除会话的响应。"""

    session_id: str = Field(description="被删除的会话 ID")
    deleted: bool = Field(
        description=(
            "会话当前是否处于已删除状态。重复删除同一个会话仍返回 true"
            "（幂等），因此 true 不表示「本次请求删掉了一行」"
        )
    )


class MessageItem(BaseModel):
    """一条消息的对外表示（用于对话内容历史）。"""

    message_id: str = Field(description="消息 ID")
    role: str = Field(description="角色：user / assistant / system")
    content: str = Field(description="消息正文")
    timestamp: Optional[str] = Field(None, description="写入时间（ISO 8601）")
    retrieved_chunks: Optional[list[dict]] = Field(
        None,
        description="该轮回答引用的检索片段快照（仅 assistant 消息有值）",
    )
