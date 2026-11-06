"""认证相关的对外数据模型。

当前只有一个接口 GET /auth/me，返回「这个 token 是谁」。
它的用途不是鉴权本身（鉴权在中间件里已经做完），
而是让前端能显示「张三 · 综合管理部」这类身份信息，
而不是只拿到一个 user_id 字符串。

display_name 与 department 允许为 null：
它们来自 token payload 里的可选字段，token 里没有就只能是 null。
不能因为缺展示名就判认证失败——身份的可信性由签名保证，
展示信息的完整度由签发方决定。

不返回剩余额度：算剩余额度就得走一次限流记账，
那会平白消耗调用方的请求次数——「查一下我还剩几次」反而少一次，
语义上是错的。
"""

from typing import Optional

from pydantic import BaseModel, Field


class CurrentUser(BaseModel):
    """当前调用方的身份。"""

    user_id: str = Field(description="全系统唯一的用户标识，取自 token")
    display_name: Optional[str] = Field(
        None,
        description="展示名，可能含中文；token 未提供时为 null",
    )
    department: Optional[str] = Field(
        None,
        description="所属部门，可能为空；token 未提供时为 null",
    )
    rate_limit_per_minute: int = Field(
        0,
        description="该用户每分钟的请求上限，0 表示未启用限流",
    )
    is_admin: bool = Field(
        False,
        description="是否具备文档管理权限（上传/删除）。前端据此决定是否渲染入口",
    )
