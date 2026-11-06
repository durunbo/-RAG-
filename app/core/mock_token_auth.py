"""测试阶段的模拟 Token 认证。

只做字符串精确匹配，不做任何解析、验签或 Claims 校验：

    Authorization: Bearer <token>  ->  token 表精确匹配  ->  user_id

这是「认证实现」的第一版。接口刻意与真实实现同形
（verify() 要么返回 AuthPrincipal，要么抛 TokenError / TokenAuthNotConfigured），
将来接入真实 Token 解析库时只替换本文件，中间件、
聊天、会话与消息存储的代码一律不动。

生产环境禁用：resources.py 在 app_env == "production"
且 auth_provider == "mock" 时直接启动失败。
"""

from dataclasses import dataclass


class TokenError(Exception):
    """token 无效：匹配不到任何用户。"""


class TokenAuthNotConfigured(RuntimeError):
    """token 表为空：部署问题，不是调用方的问题。"""


@dataclass(frozen=True)
class AuthPrincipal:
    """认证通过后的调用方身份。

    字段名与 /auth/me（CurrentUser）对齐：
    display_name / department 在模拟阶段恒为 None。
    """

    user_id: str
    display_name: str | None = None
    department: str | None = None


def parse_mock_tokens(text: str) -> dict[str, str]:
    """把 "token1:user_001, token2:user_002" 解析成 token -> user_id 映射。

    规则：
    - 逗号分隔，项内空白忽略，空项忽略；
    - 每项必须形如 "token:user_id"，缺失或为空直接抛 ValueError；
    - token 重复定义抛 ValueError（fail-fast，避免配置笔误静默覆盖）。
    """
    token_map: dict[str, str] = {}

    if not text or not text.strip():
        return token_map

    for index, pair in enumerate(text.split(","), start=1):
        pair = pair.strip()
        if not pair:
            continue

        if ":" not in pair:
            raise ValueError(f"模拟 token 第 {index} 项缺少冒号分隔: {pair!r}")

        token, user_id = pair.split(":", 1)
        token = token.strip()
        user_id = user_id.strip()

        if not token or not user_id:
            raise ValueError(f"模拟 token 第 {index} 项 token 或 user_id 为空: {pair!r}")

        if token in token_map:
            raise ValueError(f"模拟 token 重复定义: {token!r}")

        token_map[token] = user_id

    return token_map


class MockTokenAuthenticator:
    """token 精确匹配表。

    线程安全：构造后只读，verify 不做任何可变状态更新。
    """

    def __init__(self, token_map: dict[str, str]) -> None:
        self._token_map = dict(token_map)

    def verify(self, token: str) -> AuthPrincipal:
        if not self._token_map:
            raise TokenAuthNotConfigured("模拟 token 表为空，认证未配置")

        user_id = self._token_map.get(token)
        if user_id is None:
            raise TokenError("未知 token")

        return AuthPrincipal(user_id=user_id)
