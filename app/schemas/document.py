"""文档管理接口的对外数据模型。

状态与阶段用 Literal 而不是 Enum：OpenAPI 会直接列出可选值，前端能据此写类型，
后端加了新状态也不会让旧前端静默显示成 unknown。
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

IngestStatusLiteral = Literal[
    "processing",
    "succeeded",
    "needs_review",
    "skipped",
    "failed",
]

IngestStageLiteral = Literal[
    "queued",
    "parsing",
    "cleaning",
    "classifying",
    "chunking",
    "indexing",
    "finished",
]


class DocumentUploadResponse(BaseModel):
    """上传受理结果（202）。

    202 而不是 201：文件只是收下了，块还没写进库。
    """

    task_id: str = Field(description="任务 ID，用于轮询进度")
    status: IngestStatusLiteral = Field(description="任务状态，受理后恒为 processing")
    stage: IngestStageLiteral = Field(description="当前阶段")
    source_file: str = Field(description="服务端清洗后实际使用的文件名")
    created_at: datetime = Field(description="任务创建时间")
    poll_url: str = Field(description="轮询地址")


class DocumentTaskResponse(BaseModel):
    """任务状态查询结果。"""

    task_id: str
    status: IngestStatusLiteral
    stage: IngestStageLiteral = Field(description="最近一次真实执行到的阶段")
    failed_stage: Optional[IngestStageLiteral] = Field(
        None, description="失败发生在哪个阶段，仅 status=failed 时有值"
    )
    source_file: str
    document_type: Optional[str] = Field(None, description="分类得到的主类型")
    chunks_submitted: int = Field(0, description="提交写入的块数")
    chunks_inserted: int = Field(0, description="实际写入数据库的块数")
    error_code: Optional[str] = Field(None, description="失败原因码")
    message: Optional[str] = Field(None, description="可直接展示给用户的说明")
    created_at: datetime
    updated_at: datetime


class DocumentDeleteResponse(BaseModel):
    """删除结果。"""

    source_file: str
    deleted_chunks: int = Field(description="实际删除的块数")
