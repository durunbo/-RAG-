"""通用响应模型定义。"""

from typing import Optional, Generic, TypeVar
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应。

    四个字段恒定存在（details 可能为 null），前端只需按一种形状解析，
    不必再区分「中间件返回的 detail 是对象」还是「业务异常 detail 是字符串」。
    """
    
    error_code: str = Field(..., description="错误码，如 NOT_FOUND / VALIDATION_ERROR")
    message: str = Field(..., description="可直接展示给用户的错误描述")
    request_id: Optional[str] = Field(None, description="请求追踪 ID，与响应头 X-Request-ID 一致")
    details: Optional[list[dict]] = Field(
        None,
        description="字段级错误明细，仅参数校验失败（422）时非空",
    )


T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应基类。"""
    
    items: list[T] = Field(..., description="数据列表")
    total: int = Field(..., description="总记录数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, le=100, description="每页大小")
    has_next: bool = Field(..., description="是否有下一页")
