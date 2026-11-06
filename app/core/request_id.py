"""请求 ID 工具。

放在 core 而不是 api：认证中间件、请求上下文中间件和路由层都要用它，
让 middleware 反向 import app.api 会把整个 API 层拖进中间件，
而 API 层将来又会依赖 middleware，形成分层倒置与潜在循环导入。
"""

import re
import uuid
from typing import Optional


def safe_request_id(raw: Optional[str]) -> str:
    """生成安全的请求 ID。

    客户端可以自带 X-Request-ID，但那是不可信输入：
    允许换行符就能伪造日志行，允许超长就能撑爆日志。
    因此只保留 [A-Za-z0-9_.-]，并限制 32 字符。
    """
    if not raw:
        return uuid.uuid4().hex[:8]
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", raw)[:32]
    return cleaned or uuid.uuid4().hex[:8]
