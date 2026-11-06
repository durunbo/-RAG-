from pydantic import BaseModel, ConfigDict, Field


# ==========================================================
# 1. 检索请求
# ==========================================================

class RetrievalRequest(BaseModel):
    """知识检索接口的请求参数。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="用户提出的检索问题",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最多返回的检索结果数量",
    )


# ==========================================================
# 2. 来源信息
# ==========================================================

class RetrievalSource(BaseModel):
    """知识片段的来源和业务元数据。"""

    # 忽略向量库中未对外声明的额外字段
    model_config = ConfigDict(extra="ignore")

    source_file: str = Field(
        description="原始文档名称",
    )

    document_type: str | None = Field(
        default=None,
        description="文档类型",
    )

    heading_path: str | None = Field(
        default=None,
        description="文档章节路径",
    )

    chunk_index: int | None = Field(
        default=None,
        ge=0,
        description="Chunk 在原始文档中的序号",
    )

    source_page: int | None = Field(
        default=None,
        description="原始文档页码",
    )

    question: str | None = Field(
        default=None,
        description="FAQ 对应的问题",
    )

    section: str | None = Field(
        default=None,
        description="所属章节",
    )

    stage: str | None = Field(
        default=None,
        description="流程阶段",
    )

    responsible_department: str | None = Field(
        default=None,
        description="责任部门",
    )

    form_name: str | None = Field(
        default=None,
        description="关联表单名称",
    )


# ==========================================================
# 3. 单条检索结果
# ==========================================================

class RetrievalItem(BaseModel):
    """单条命中的知识片段。"""

    score: float = Field(
        description="向量检索相似度分数，不代表答案正确率",
    )

    text: str = Field(
        description="命中的知识片段正文",
    )

    source: RetrievalSource = Field(
        description="知识片段的来源信息",
    )

    @classmethod
    def from_raw(cls, raw: dict) -> "RetrievalItem":
        """把向量库返回的原始字典转成对外契约。

        「空字符串表示该字段不存在」是向量库的存储约定，转换只在这一处做。
        以前 chat 和 retrieval 两个路由各写一份，retrieval 那份漏了
        "" → None 的清洗，页码这类整数字段收到空串直接抛 int_parsing，
        /retrieval/search 因此长期返回 500。
        """
        if not isinstance(raw, dict):
            raise TypeError("检索结果不是有效对象")

        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError("检索结果缺少有效的 metadata")

        cleaned_metadata = {
            key: (None if value == "" else value)
            for key, value in metadata.items()
        }

        return cls(
            score=float(raw["score"]),
            text=raw["text"],
            source=RetrievalSource.model_validate(cleaned_metadata),
        )


# ==========================================================
# 4. 检索响应
# ==========================================================

class RetrievalResponse(BaseModel):
    """知识检索接口的统一响应。"""

    query: str = Field(
        description="本次执行的检索问题",
    )

    requested_top_k: int = Field(
        ge=1,
        le=20,
        description="请求的最大结果数量",
    )

    returned_count: int = Field(
        ge=0,
        description="实际返回的结果数量",
    )

    results: list[RetrievalItem] = Field(
        description="检索结果列表",
    )