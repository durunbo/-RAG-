import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.services.llm_client import DEFAULT_PING_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)

_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


class OllamaClientError(RuntimeError):
    """Ollama 调用失败时抛出的统一异常。"""


class OllamaClient:
    """
    Ollama 本地大模型客户端。

    负责：
    1. 调用 Ollama /api/chat 接口。
    2. 校验输入消息。
    3. 处理网络、HTTP 和响应格式异常。
    4. 返回模型最终回答文本。

    不负责：
    1. 向量检索。
    2. RAG 提示词构造。
    3. 会话存储和权限校验。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        temperature: float = 0.2,
        num_predict: int = 400,
    ) -> None:
        # 一、校验服务地址
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama API 地址不能为空")

        normalized_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError(
                "Ollama API 地址必须是有效的 HTTP 或 HTTPS URL"
            )

        # 二、校验模型名称
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama 模型名称不能为空")

        # 三、校验超时时间
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Ollama 请求超时时间必须是有限的正数")

        # 四、校验生成参数
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not (0.0 <= float(temperature) <= 2.0)
        ):
            raise ValueError("temperature 必须是 0~2 之间的数字")

        if (
            isinstance(num_predict, bool)
            or not isinstance(num_predict, int)
            or (num_predict <= 0 and num_predict != -1)
        ):
            raise ValueError("num_predict 必须是正整数（-1 表示不限制）")

        self.base_url = normalized_url
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)

    def _resolve_temperature(self, override: Optional[float]) -> float:
        """解析本次调用的温度。不传沿用构造值，传了就必须是合法温度。

        单个参数覆盖而不是每次新建客户端：客户端是按 provider 装配一次的，
        回答生成希望保留一点自然度（0.2），而抽取/对齐必须是 0。
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

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        format: Optional[dict] = None,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        发送聊天消息，返回模型的最终回答。

        示例：
        [
            {"role": "system", "content": "你是企业知识助手。"},
            {"role": "user", "content": "请总结这段制度。"},
        ]

        Args:
            format: 可选的 JSON Schema。传入时 Ollama 用它构建解码语法，
                输出被约束为符合该结构的 JSON 文本（不保证语义正确，
                调用方仍需校验内容）。不传则维持原来的自由文本行为。
            num_predict: 单次调用的输出上限，不传则用构造时的默认值。
                结构化抽取的输出比自然语言回答长，需要单独放宽，
                否则会在记录中途被截断、产出半截 JSON。
            temperature: 单次调用的采样温度，不传则用构造时的默认值。
                结构化判定/抽取必须传 0：默认 0.2 仍带随机性，
                同一问题会出现「这次判同一件事、下次判无关」的抖动。
        """

        # 一、校验消息列表
        if (
            isinstance(messages, (str, bytes))
            or not isinstance(messages, Sequence)
            or not messages
        ):
            raise ValueError("messages 必须是非空消息列表")

        normalized_messages: list[dict[str, str]] = []

        for index, message in enumerate(messages, start=1):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"第 {index} 条消息必须是对象"
                )

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or role not in _ALLOWED_ROLES:
                raise ValueError(
                    f"第 {index} 条消息的 role 不合法"
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"第 {index} 条消息的 content 必须是字符串"
                )

            normalized_messages.append({
                "role": role,
                "content": content,
            })

        # 二、构造 Ollama 请求
        effective_num_predict = self.num_predict
        if num_predict is not None:
            if (
                isinstance(num_predict, bool)
                or not isinstance(num_predict, int)
                or (num_predict <= 0 and num_predict != -1)
            ):
                raise ValueError("num_predict 必须是正整数（-1 表示不限制）")
            effective_num_predict = int(num_predict)

        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
            # 显式传入生成参数，避免回落到 Ollama 默认值（temperature=0.8、
            # num_predict=-1 不限制输出长度，会让答案发散且延迟不可控）
            "options": {
                "temperature": self._resolve_temperature(temperature),
                "num_predict": effective_num_predict,
            },
        }

        # 结构化输出：format 传 JSON Schema 时，Ollama 会据此构建解码语法，
        # 模型只能按该结构吐字。注意它只约束结构，不约束事实 —— 内容对错
        # 仍由调用方校验（见 rag_service 的证据校验）。
        if format is not None:
            if not isinstance(format, dict):
                raise ValueError("format 必须是 JSON Schema 对象（dict）")
            payload["format"] = format

        try:
            request_data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("无法序列化 Ollama 请求") from exc

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=request_data,
            headers={
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        # 三、调用 Ollama API
        logger.info(f"调用 Ollama: model={self.model}, messages_count={len(normalized_messages)}")
        
        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_body = response.read().decode("utf-8")
                logger.info(f"Ollama 响应成功，长度: {len(raw_body)}")

        except HTTPError as exc:
            status_code = exc.code
            exc.close()

            raise OllamaClientError(
                f"Ollama API 返回 HTTP {status_code}"
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaClientError(
                "无法连接 Ollama，或请求超时"
            ) from exc

        except UnicodeDecodeError as exc:
            raise OllamaClientError(
                "Ollama 返回的响应不是有效的 UTF-8 文本"
            ) from exc

        # 四、解析并校验 JSON
        try:
            result = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise OllamaClientError(
                "Ollama 返回了无效的 JSON 响应"
            ) from exc

        if not isinstance(result, dict):
            raise OllamaClientError(
                "Ollama 响应格式错误：顶层必须是 JSON 对象"
            )

        # 五、提取最终回答，不返回 thinking 字段
        message = result.get("message")

        if not isinstance(message, dict):
            raise OllamaClientError(
                "Ollama 响应缺少有效的 message 对象"
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise OllamaClientError(
                "Ollama 没有返回有效的最终回答"
            )

        return content.strip()

    # ----------------------------------------
    # 探活：供 /health/ready 使用
    # ----------------------------------------
    @staticmethod
    def _model_matches(installed: str, configured: str) -> bool:
        """判断已安装模型名是否匹配配置的模型名。

        Ollama 的 /api/tags 返回完整名（如 qwen2.5:7b）。
        配置里省略 tag 时等价于 :latest，需要容忍这种写法；
        但两边都写了 tag 就必须完全一致（qwen2.5:7b ≠ qwen2.5:14b）。
        """
        if installed == configured:
            return True

        base_installed, _, tag_installed = installed.partition(":")
        base_configured, _, tag_configured = configured.partition(":")

        if base_installed != base_configured:
            return False

        return tag_installed == "" or tag_configured == ""

    def ping(self) -> None:
        """探活：GET /api/tags，确认服务可达且配置的模型已拉取。

        查模型是否存在而不只是查 HTTP 200，是因为最常见的故障不是 Ollama 没起来，
        而是配置写了个本机没 pull 的模型名——那种情况下 /api/tags 依然是 200，
        只有真正发一次 chat 才会失败。
        """
        endpoint = f"{self.base_url}/api/tags"
        request = Request(url=endpoint, method="GET")

        try:
            with urlopen(
                request,
                timeout=DEFAULT_PING_TIMEOUT_SECONDS,
            ) as response:
                raw_body = response.read().decode("utf-8")

        except HTTPError as exc:
            status_code = exc.code
            exc.close()

            raise OllamaClientError(
                f"Ollama 探活失败（HTTP {status_code}）"
            ) from None

        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaClientError(
                f"无法连接 Ollama：{endpoint}"
            ) from exc

        except UnicodeDecodeError as exc:
            raise OllamaClientError(
                "Ollama 探活响应不是有效的 UTF-8 文本"
            ) from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise OllamaClientError(
                "Ollama 探活返回了无效的 JSON 响应"
            ) from exc

        models = body.get("models") if isinstance(body, dict) else None

        if not isinstance(models, list):
            raise OllamaClientError("Ollama 探活响应缺少 models 列表")

        installed: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            for key in ("name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value and value not in installed:
                    installed.append(value)

        if not any(
            self._model_matches(name, self.model) for name in installed
        ):
            raise OllamaClientError(
                f"Ollama 未安装模型 {self.model}"
                f"（已安装：{installed or '无'}），"
                f"请先执行 ollama pull {self.model}"
            )
