"""
会话与消息数据模型定义。

对应 PostgreSQL 表结构：
- sessions: 会话元数据（user_id、创建时间等）
- messages: 对话消息记录（含检索结果快照）
"""

from sqlalchemy import Column, String, Text, DateTime, BigInteger, Index, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SessionModel(Base):
    """会话表：存储用户会话元数据。

    逻辑删除：删除会话只写 deleted_at，消息记录全部保留。
    所有面向业务的查询都必须带 deleted_at IS NULL，否则已删除的会话
    会重新出现在列表里、或继续被当作聊天上下文读出来。
    """

    __tablename__ = "sessions"

    session_id = Column(String(64), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column(JSONB, nullable=True)  # 扩展字段（如会话标题、标签等）
    # NULL = 未删除。刻意用时间戳而不是 boolean：
    # 既能表达「是否删除」，也能支撑将来「按删除时间定期物理清理」，
    # 而 boolean 一旦需要时间就得再加一列。
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # 列表查询专用部分索引，必须与 scripts/migrations/003 的定义逐字一致，
    # 否则 ORM 建出来的库和 SQL 建出来的库索引不同，排查会非常痛苦。
    __table_args__ = (
        Index(
            "idx_sessions_user_active",
            "user_id",
            "updated_at",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )


class MessageModel(Base):
    """消息表：存储对话历史及检索结果快照。"""

    __tablename__ = "messages"

    message_id = Column(String(64), primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # 'user' / 'assistant' / 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    retrieved_chunks = Column(JSONB, nullable=True)  # 检索到的文档片段（仅 assistant 消息有值）
