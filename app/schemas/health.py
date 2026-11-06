from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """健康检查接口的统一响应模型。"""

    status: Literal["alive", "ready", "not_ready"]
    service: str
    version: str