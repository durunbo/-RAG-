"""从调用方 token 解析身份。

为什么是这个方案，而不是服务端自己发 Key：

本服务前面已经有一套登录系统。用户登录后拿到 token，请求时带上即可。
如果这里再让运维给每个人发一把 API Key，等于凭空造出第二套身份体系：
token 说「这人是张三」，api_keys 表说「这人是 user_001」，
两套平时一致，一旦不一致（人调部门、离职没跑脚本、发错人）就说不清以谁为准。
而且要人记 43 字符的串、丢了重发、离职要跑吊销——这些成本全是白加的。

所以身份的唯一来源是 token，本服务不维护任何用户名录。

两条必须守住的底线：

1. **不验签就等于没有认证。**
   JWT 的 payload 只是 base64url，任何人都能改。只有签名验证通过，
   「这人是张三」才成立。所以本模块没有任何「跳过验签」的分支：
   没配密钥时构造函数直接抛错，服务起不来，而不是默默放行。

2. **只接受配置里指定的算法，绝不信任 header 里的 alg。**
   这是 JWT 最经典的漏洞（alg=none / 算法混淆）：攻击者把 header 改成
   {"alg":"none"} 并去掉签名，宽松的实现就会直接采信 payload。
   这里要求 header.alg 必须等于配置值，否则一律拒绝。

零新增依赖：HS256/384/512 用标准库 hmac + hashlib 即可，
不需要为了验签引入 PyJWT 或 cryptography。
如果对接方最后给的是 RS256 公钥，届时需要安装 cryptography——那时再加。
"""

import base64
import binascii
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

# 支持的 HMAC 算法。故意不包含 "none"：那不是一个算法，那是一扇后门。
_SUPPORTED_ALGORITHMS: dict[str, callable] = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}

# token 明文长度上限。正常 JWT 几百字符，超过这个量级一定是恶意构造。
_MAX_TOKEN_LENGTH = 8192


# ── 异常：按「谁的问题」分开，中间件据此决定状态码 ──────────────


class TokenError(Exception):
    """所有 token 校验失败的基类。"""


class TokenMalformed(TokenError):
    """结构不对：不是三段、base64 解不开、不是 JSON 对象。"""


class TokenAlgorithmRejected(TokenError):
    """header.alg 与配置不符，或用了不支持的算法（含 none）。"""


class TokenSignatureInvalid(TokenError):
    """签名对不上：token 被改过，或者不是用这把密钥签的。"""


class TokenExpired(TokenError):
    """exp 已过，或 nbf 还没到。"""


class TokenClaimMissing(TokenError):
    """签名没问题，但取不到用户标识字段。"""


class TokenAuthNotConfigured(RuntimeError):
    """服务端没配密钥。

    这不是调用方的错，所以不能返回 401 让人去改请求——
    它应该表现为 503，并在日志里明确指向配置项。
    """


# ── 身份 ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthPrincipal:
    """验签通过后的调用方身份。

    frozen 是为了防止下游路由改写 user_id：
    身份一旦被中途替换，后面所有归属校验都会失效。

    display_name / department 允许为空——它们取决于 token 里带了什么字段，
    而字段由签发方决定，本服务管不着。不能因为缺个部门就拒绝认证。
    """

    user_id: str
    display_name: str | None = None
    department: str | None = None


# ── base64url 与 claim 取值 ─────────────────────────────────


def _b64url_decode(segment: str) -> bytes:
    """严格解码 base64url。

    不用 urlsafe_b64decode 直接解：它遇到非法字符会静默忽略，
    于是 "abc$def" 和 "abcdef" 解出同样的结果。
    验签场景下这种宽容是危险的——它让畸形输入有机会伪装成合法输入。
    """
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.b64decode(
            segment + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise TokenMalformed("token 段不是合法的 base64url") from exc


def _json_object(raw: bytes, what: str) -> dict:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TokenMalformed(f"token 的 {what} 不是合法 JSON") from exc

    if not isinstance(value, dict):
        raise TokenMalformed(f"token 的 {what} 不是 JSON 对象")

    return value


def _get_claim(payload: dict, path: str):
    """按点号路径取 claim，例如 "user.id"。

    支持嵌套是因为不同签发方的写法差别很大：
    有的把账号放 sub，有的放 user_id，有的塞进 user.account。
    做成配置项，对接时改 .env 即可，不用改代码。
    """
    if not path:
        return None

    current = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None

    return current


# ── 验证器 ──────────────────────────────────────────────────


class TokenAuthenticator:
    """验签并取出调用方身份。无状态，可跨请求复用。"""

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        issuer: str = "",
        audience: str = "",
        leeway_seconds: int = 0,
        require_exp: bool = True,
        uid_claim: str = "sub",
        name_claim: str = "",
        department_claim: str = "",
    ) -> None:
        """
        Args:
            secret: HS256 的共享密钥。不能为空。
            algorithm: 期望算法，必须是 _SUPPORTED_ALGORITHMS 里的一个。
            issuer: 期望的 iss，空串表示不校验。
            audience: 期望的 aud，空串表示不校验。
            leeway_seconds: 时钟偏移容忍秒数，避免两端时间差导致误判过期。
            require_exp: 是否要求 token 必须带 exp。
                默认 True。没有 exp 的 token 永不过期，一旦泄漏就长期可用，
                所以默认拒绝；确需放宽时显式配置。
            uid_claim: 用户标识所在的 claim 路径。
            name_claim: 展示名所在的 claim 路径，空串表示不读。
            department_claim: 部门所在的 claim 路径，空串表示不读。
        """
        if not secret:
            # 这里必须抛错而不是警告后继续：
            # 没有密钥就无法验签，而「无法验签」的实现等于任何人都能
            # 自己编一个 token 冒充别人读别人的会话。宁可服务起不来。
            raise ValueError(
                "未配置 token 验签密钥（RAG_AUTH_JWT_SECRET）。"
                "认证不能在没有密钥的情况下运行。"
            )

        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"不支持的签名算法：{algorithm}。"
                f"可选：{', '.join(sorted(_SUPPORTED_ALGORITHMS))}。"
                "注意 alg=none 永远不被接受。"
            )

        self._secret = secret.encode("utf-8")
        self._algorithm = algorithm
        self._hash = _SUPPORTED_ALGORITHMS[algorithm]
        self._issuer = issuer
        self._audience = audience
        self._leeway = max(0, int(leeway_seconds))
        self._require_exp = require_exp
        self._uid_claim = uid_claim
        self._name_claim = name_claim
        self._department_claim = department_claim

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def verify(self, token: str) -> AuthPrincipal:
        """验证 token 并返回身份。

        Raises:
            TokenMalformed / TokenAlgorithmRejected / TokenSignatureInvalid /
            TokenExpired / TokenClaimMissing: 各类校验失败。
            调用方（中间件）统一翻译成 401。
        """
        if not token or len(token) > _MAX_TOKEN_LENGTH:
            raise TokenMalformed("token 为空或过长")

        segments = token.split(".")
        if len(segments) != 3:
            raise TokenMalformed("token 不是三段式 JWT")

        header_segment, payload_segment, signature_segment = segments

        # ── 一、先验算法，再验签名 ──────────────────────────
        # 顺序很关键：如果先解 payload 再验签，中间任何一步采信了未验证的数据，
        # 都等于给了攻击者一个可操作的面。
        header = _json_object(_b64url_decode(header_segment), "header")

        token_algorithm = header.get("alg")
        if not isinstance(token_algorithm, str):
            raise TokenAlgorithmRejected("token header 缺少 alg")

        # 必须与配置完全一致。这一步同时挡掉 alg=none 和算法混淆：
        # 攻击者把 HS256 改成 none 或 RS256 都会在这里被拒。
        if token_algorithm != self._algorithm:
            raise TokenAlgorithmRejected(
                f"算法不符：期望 {self._algorithm}，实际 {token_algorithm}"
            )

        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        expected_signature = hmac.new(
            self._secret,
            signing_input,
            self._hash,
        ).digest()

        actual_signature = _b64url_decode(signature_segment)

        # compare_digest 是常数时间比较：普通 == 会在第一个不同字节就返回，
        # 攻击者能靠响应时间差逐字节猜出正确签名。
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise TokenSignatureInvalid("token 签名校验失败")

        # ── 二、签名通过后，payload 才可信 ──────────────────
        payload = _json_object(_b64url_decode(payload_segment), "payload")

        self._check_time(payload)
        self._check_issuer(payload)
        self._check_audience(payload)

        user_id = _get_claim(payload, self._uid_claim)
        if user_id is None or user_id == "":
            raise TokenClaimMissing(
                f"token 里没有用户标识字段（当前配置：{self._uid_claim}）"
            )

        # 有的签发方把账号写成数字，统一转成字符串，
        # 免得后面拿 user_id 去查库时类型对不上
        user_id = str(user_id)

        display_name = _get_claim(payload, self._name_claim)
        department = _get_claim(payload, self._department_claim)

        return AuthPrincipal(
            user_id=user_id,
            display_name=str(display_name) if display_name is not None else None,
            department=str(department) if department is not None else None,
        )

    # ── 内部校验 ────────────────────────────────────────────

    def _check_time(self, payload: dict) -> None:
        now = time.time()

        exp = payload.get("exp")
        if exp is None:
            if self._require_exp:
                raise TokenExpired(
                    "token 没有 exp，拒绝永久有效的凭据"
                    "（如确需放宽，配置 RAG_AUTH_REQUIRE_EXP=false）"
                )
        elif not isinstance(exp, (int, float)) or isinstance(exp, bool):
            raise TokenMalformed("exp 不是数字")
        elif now > exp + self._leeway:
            raise TokenExpired("token 已过期")

        nbf = payload.get("nbf")
        if isinstance(nbf, (int, float)) and not isinstance(nbf, bool):
            if now + self._leeway < nbf:
                raise TokenExpired("token 尚未生效（nbf 在未来）")

    def _check_issuer(self, payload: dict) -> None:
        if not self._issuer:
            return

        if payload.get("iss") != self._issuer:
            raise TokenClaimMissing(
                f"iss 不符：期望 {self._issuer}，实际 {payload.get('iss')}"
            )

    def _check_audience(self, payload: dict) -> None:
        if not self._audience:
            return

        aud = payload.get("aud")
        # aud 按规范可以是字符串或字符串数组
        if isinstance(aud, str):
            audiences = [aud]
        elif isinstance(aud, list):
            audiences = [item for item in aud if isinstance(item, str)]
        else:
            audiences = []

        if self._audience not in audiences:
            raise TokenClaimMissing(
                f"aud 不符：期望 {self._audience}，实际 {aud}"
            )
