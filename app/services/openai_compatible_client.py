"""OpenAI 兼容协议的 LLM 客户端（DeepSeek / 通义千问 等）。

与 OllamaClient 的差异全部收敛在本文件，业务代码不感知：

| 环节     | Ollama                            | OpenAI 兼容                       |
|----------|-----------------------------------|-----------------------------------|
| 路径     | POST /api/chat                    | POST {base_url}/chat/completions  |
| 鉴权     | 无                                | Authorization: Bearer <key>       |
| 生成参数 | options.{temperature,num_predict} | 顶层 temperature / max_tokens     |
| 响应取值 | body.message.content              | body.choices[0].message.content   |

base_url 该填什么（填错直接 404，这是最容易踩的坑）：
- DeepSeek  : https://api.deepseek.com            （带不带 /v1 都行）
- 通义千问  : https://dashscope.aliyuncs.com/compatible-mode/v1
- 其它自建  : 指到 /v1 这一层，本模块只追加 /chat/completions

已知不兼容（第一轮不处理，切模型时避开）：
- deepseek-reasoner 不接受 temperature，传了会 400，请用 deepseek-chat
- 不发送 num_ctx（Ollama 专有），上下文长度由服务端默认值决定
"""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.services.llm_client import DEFAULT_PING_TIMEOUT_SECONDS, LLMClientError

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})

# 错误体只取这么长的 message，避免把整段 HTML/JSON 塞进日志
_ERROR_MESSAGE_LIMIT = 200


class OpenAICompatibleClient:
    """DeepSeek / 通义千问等 OpenAI 兼容服务的同步客户端。

    与 OllamaClient 使用同一个 urllib 实现，不引入 httpx / openai SDK：
    依赖零新增，且两种实现的超时、异常、字符串处理行为完全同构，
    便于「只换 provider 不改业务」的对比验证。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float = 180.0,
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> None:
        # 一、校验服务地址
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("LLM API 地址不能为空")

        normalized_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError("LLM API 地址必须是有效的 HTTP 或 HTTPS URL")

        # 二、校验模型名称
        if not isinstance(model, str) or not model.strip():
            raise ValueError("LLM 模型名称不能为空")

        # 三、校验凭据。
        # 这是与 Ollama 最大的差别：云端必须有 key。
        # 在这里报错 = 启动即失败，而不是等用户提问时才 401。
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "provider=openai_compatible 时必须配置 RAG_LLM_API_KEY"
            )

        # 四、校验超时时间
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("LLM 请求超时时间必须是有限的正数")

        # 五、校验生成参数
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not (0.0 <= float(temperature) <= 2.0)
        ):
            raise ValueError("temperature 必须是 0~2 之间的数字")

        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens 必须是正整数")

        self.base_url = normalized_url
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        # 凭据不留明文属性，日志里只出现后 4 位，便于确认「配的是哪一把 key」
        self._api_key = api_key.strip()
        self._api_key_tail = self._api_key[-4:]

    # ----------------------------------------
    # 内部
    # ----------------------------------------
    def _resolve_temperature(self, override: float | None) -> float:
        """解析本次调用的温度。不传沿用构造值，传了就必须是合法温度。

        加这个覆盖口子是因为默认 0.2 仍带随机性：同一问题两次可能一次判
        「同一件事」一次判「无关」。结构化判定与条款抽取必须固定 0，
        而回答生成保留 0.2 的自然度，两者不能共用一个常量。
        """
        if override is None:
            return self.temperature
        if (
            isinstance(override, bool)
            or not isinstance(override, (int, float))
            or not math.isfinite(override)
            or not (0.0 <= float(override) <= 2.0)
        ):
            raise ValueError("temperature 必须是 0~2 之间的数字")
        return float(override)

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        """校验并规范化消息列表。

        规则与 OllamaClient.chat() 内的校验保持一致。第一轮刻意重复一份而不是
        抽公共函数：抽的话要改已验证的 ollama_client.py，与本次「只做模型抽象」
        的范围冲突。等做健康检查那一批时再合并。
        """
        if (
            isinstance(messages, (str, bytes))
            or not isinstance(messages, Sequence)
            or not messages
        ):
            raise ValueError("messages 必须是非空消息列表")

        normalized: list[dict[str, str]] = []

        for index, message in enumerate(messages, start=1):
            if not isinstance(message, Mapping):
                raise ValueError(f"第 {index} 条消息必须是对象")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or role not in _ALLOWED_ROLES:
                raise ValueError(f"第 {index} 条消息的 role 不合法")

            if not isinstance(content, str):
                raise ValueError(f"第 {index} 条消息的 content 必须是字符串")

            normalized.append({"role": role, "content": content})

        return normalized

    @staticmethod
    def _extract_error_detail(exc: HTTPError) -> str:
        """从 OpenAI 风格错误体 {"error": {"message": ...}} 里提取一句话。

        非 JSON 错误体（网关 HTML、空响应）返回空串，只回状态码，
        避免把整段 HTML 或内部地址塞进日志与响应。
        """
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    return str(error.get("message") or "")[:_ERROR_MESSAGE_LIMIT]
                if isinstance(error, str):
                    return error[:_ERROR_MESSAGE_LIMIT]
        except Exception:  # noqa: BLE001 - 解析失败不应掩盖原始 HTTP 错误
            pass
        finally:
            exc.close()

        return ""

    # ----------------------------------------
    # 对外：与 OllamaClient 完全一致的同步契约
    # ----------------------------------------
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        format: dict | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """发送聊天消息，返回模型的最终回答。

        Args:
            format: 可选 JSON Schema。这里**只能降级为 JSON 模式**
                （response_format={"type": "json_object"}）：
                DeepSeek 等服务不支持 json_schema 结构化输出，
                直接下发会 400。因此结构约束由 rag_service 的本地校验兜底
                —— 校验不通过的记录一律丢弃，不会进入答案。
            num_predict: 单次调用的输出上限，不传则用构造时的默认值。
                契约沿用 Ollama 的命名（见 llm_client 的说明），
                这里映射为 OpenAI 规范的 max_tokens。
            temperature: 单次调用的采样温度，不传则用构造时的默认值。
                结构化判定/抽取必须传 0，否则同一问题会出现判定抖动。
        """
        normalized_messages = self._normalize_messages(messages)

        effective_max_tokens = self.max_tokens
        if num_predict is not None:
            if (
                isinstance(num_predict, bool)
                or not isinstance(num_predict, int)
                or num_predict <= 0
            ):
                raise ValueError("num_predict 必须是正整数")
            effective_max_tokens = int(num_predict)

        # 注意与 Ollama 的差别：生成参数在顶层，没有 options 嵌套。
        # 只发 OpenAI 规范内确定的字段，不发各家私有扩展（如 Qwen 的
        # enable_search），避免换供应商时静默失效。
        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": effective_max_tokens,
        }

        if format is not None:
            if not isinstance(format, dict):
                raise ValueError("format 必须是 JSON Schema 对象（dict）")
            payload["response_format"] = {"type": "json_object"}

        try:
            request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("无法序列化 LLM 请求") from exc

        endpoint = f"{self.base_url}/chat/completions"

        request = Request(
            url=endpoint,
            data=request_data,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        logger.info(
            "调用 LLM(openai_compatible): model=%s messages_count=%s key尾号=%s",
            self.model,
            len(normalized_messages),
            self._api_key_tail,
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")

        except HTTPError as exc:
            status_code = exc.code
            detail = self._extract_error_detail(exc)

            # 401/403 与 429 单独区分：这两类是配置或配额问题，
            # 与「模型答不出来」的处理方式完全不同，报错文案必须能指向配置项。
            if status_code in (401, 403):
                raise LLMClientError(
                    f"LLM 鉴权失败（HTTP {status_code}），请检查 "
                    "RAG_LLM_API_KEY 是否正确、是否过期"
                ) from None

            if status_code == 429:
                raise LLMClientError(
                    "LLM 服务限流（HTTP 429），请稍后重试"
                ) from None

            raise LLMClientError(
                f"LLM 服务返回 HTTP {status_code}"
                + (f"：{detail}" if detail else "")
            ) from None

        except (URLError, TimeoutError, OSError) as exc:
            # 只带 key 尾号与地址，不带完整 key
            raise LLMClientError(
                f"无法连接 LLM 服务 {endpoint}"
                f"（key 尾号 {self._api_key_tail}），或请求超时"
            ) from exc

        except UnicodeDecodeError as exc:
            raise LLMClientError(
                "LLM 返回的响应不是有效的 UTF-8 文本"
            ) from exc

        # 解析并校验 JSON
        try:
            result = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM 返回了无效的 JSON 响应") from exc

        if not isinstance(result, dict):
            raise LLMClientError(
                "LLM 响应格式错误：顶层必须是 JSON 对象"
            )

        choices = result.get("choices")

        # 内容审核或限流时部分服务会返回空 choices。
        # 必须显式报错，否则 choices[0] 抛 IndexError 变成难排查的 500。
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("LLM 响应缺少 choices")

        first = choices[0]
        if not isinstance(first, dict):
            raise LLMClientError("LLM 响应的 choices[0] 不是对象")

        if first.get("finish_reason") == "length":
            # 与 Ollama 侧行为保持一致：不报错，只记日志。
            # 出现它说明该调大 RAG_LLM_MAX_TOKENS。
            logger.warning(
                "LLM 回答被 max_tokens 截断：model=%s max_tokens=%s",
                self.model,
                self.max_tokens,
            )

        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMClientError("LLM 响应缺少有效的 message 对象")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("LLM 没有返回有效的最终回答")

        return content.strip()

    # ----------------------------------------
    # 探活：供 /health/ready 使用
    # ----------------------------------------
    def ping(self) -> None:
        """探活：GET {base_url}/models，确认服务可达、鉴权有效。

        与 Ollama 侧的差别：这里**不把「模型不在列表里」当成失败**。
        部分网关/代理的 /models 只列部分可调模型，用它判断可用性会误报未就绪，
        把一个能正常回答的服务踢出负载均衡。因此只记 warning，判定交给真实调用。
        """
        endpoint = f"{self.base_url}/models"

        request = Request(
            url=endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=DEFAULT_PING_TIMEOUT_SECONDS,
            ) as response:
                raw_body = response.read().decode("utf-8")

        except HTTPError as exc:
            status_code = exc.code
            detail = self._extract_error_detail(exc)

            if status_code in (401, 403):
                raise LLMClientError(
                    f"LLM 探活鉴权失败（HTTP {status_code}），请检查 "
                    "RAG_LLM_API_KEY 是否正确、是否过期"
                ) from None

            raise LLMClientError(
                f"LLM 探活返回 HTTP {status_code}"
                + (f"：{detail}" if detail else "")
            ) from None

        except (URLError, TimeoutError, OSError) as exc:
            raise LLMClientError(
                f"无法连接 LLM 服务 {endpoint}"
                f"（key 尾号 {self._api_key_tail}），或探活超时"
            ) from exc

        except UnicodeDecodeError as exc:
            raise LLMClientError(
                "LLM 探活响应不是有效的 UTF-8 文本"
            ) from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM 探活返回了无效的 JSON 响应") from exc

        data = body.get("data") if isinstance(body, dict) else None

        if not isinstance(data, list):
            raise LLMClientError("LLM 探活响应缺少 data 列表")

        available = [
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

        if available and self.model not in available:
            logger.warning(
                "LLM 探活成功，但配置的模型不在 /models 列表中："
                "model=%s 列表=%s（部分网关只列部分模型，此处不作为失败）",
                self.model,
                available,
            )

        logger.info(
            "LLM 探活成功：model=%s key尾号=%s",
            self.model,
            self._api_key_tail,
        )
