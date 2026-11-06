"""LLM 客户端统一契约与工厂。

三条刻意保持的边界（理由写在这里，避免后来者以为是漏了）：

1. 契约只有 chat() 与 ping() 两个方法。
   ping() 专供健康检查：Ollama 走 GET /api/tags，OpenAI 兼容走 GET /models，
   两者的探活路径与鉴权方式都不同，差异全部收敛在各自适配器内部，
   健康检查只负责调用 ping()，不感知 provider。
   ping() 失败时**抛异常而不是返回 False**：调用方需要拿到「为什么不可用」
   （鉴权过期 / 模型没拉取 / 连不上），布尔值丢掉了这些信息。

2. 同步契约。
   现有 RAGService.answer() 是 asyncio.to_thread(self.llm_client.chat, ...)，
   并发上限被 Semaphore 夹在 3，而 to_thread 默认线程池是 min(32, cpu+4)，
   同步实现完全撑得住。第一轮不同时改「调用方式」和「实现」，避免两个变量
   混在一起无法定位回归。

3. 用 Protocol 而不是 ABC。
   OllamaClient 已有 chat(self, messages) -> str，结构化类型自动匹配，
   不需要显式继承；新增 ping() 同理，只要方法名与签名对得上即可。
"""

import logging
from typing import Mapping, Protocol, Sequence, runtime_checkable

from app.core.config import Settings

logger = logging.getLogger(__name__)


# 探活超时，与生成超时（RAG_LLM_TIMEOUT_SECONDS=180s）刻意分开：
# 健康检查必须快速给结论，否则探针会长时间挂起、K8s 判定超时反而掩盖真实原因。
# 5s 对本地 Ollama（/api/tags 是纯内存查询）和云端 /models 都足够。
DEFAULT_PING_TIMEOUT_SECONDS = 5.0


class LLMClientError(RuntimeError):
    """新适配器抛出的统一异常。

    没有要求 OllamaClientError 改继承它：调用方 RAGService.answer() 统一
    catch Exception 后转 GenerationFailed，两类异常不需要合并；强行统一继承
    关系要改动已验证的 ollama_client.py，收益为零、风险不为零。
    """


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端契约：实现方需提供 chat() 与 ping()。"""

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        format: dict | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """发送消息列表，返回模型最终回答文本（已 strip）。

        Args:
            messages: [{"role": "system|user|assistant", "content": "..."}]
            format: 可选 JSON Schema，用于结构化输出。**实现方必须支持
                「不传时行为完全不变」**：Ollama 侧映射为 format，
                OpenAI 兼容侧降级为 json_object（该协议族普遍不支持
                json_schema）。schema 只约束结构，内容正确性由调用方校验。
            num_predict: 单次调用的输出上限，不传则用构造时的默认值。
                契约名沿用 Ollama 的 num_predict（与 Settings.llm_max_tokens
                是同一语义），OpenAI 兼容侧映射为 max_tokens。
                之所以要单次覆盖：结构化抽取的输出远长于自然语言回答。
            temperature: 单次调用的采样温度，不传则用构造时的默认值。
                结构化判定与抽取必须传 0 —— 默认 0.2 的随机性会让同一个
                问题出现「这次判同一件事、下次判无关」的抖动。

        Raises:
            实现方自有异常，例如 OllamaClientError / LLMClientError。
        """
        ...

    def ping(self) -> None:
        """探活：确认服务可达、鉴权有效、配置的模型可用。

        成功返回 None，**失败必须抛异常**（不要返回 False），
        以便健康检查把具体原因写进日志。

        Raises:
            实现方自有异常，例如 OllamaClientError / LLMClientError。
        """
        ...


def create_llm_client(settings: Settings) -> LLMClient:
    """按 settings.llm_provider 创建客户端。

    实现用函数内导入，避免 llm_client -> 适配器 -> llm_client 的循环依赖：
    适配器需要 import 这里的 LLMClientError。

    新增供应商的步骤：
    1. 在 config.py 的 llm_provider Literal 里加取值；
    2. 新建适配器文件，实现 chat(messages) -> str；
    3. 在下面加一个 elif 分支。

    协议不兼容的新厂商（不是 OpenAI 兼容风格）必须新增适配器，
    仅改 .env 是接不进来的——这是本设计的边界，不做过度承诺。
    """
    provider = (settings.llm_provider or "").strip().lower()

    if provider == "ollama":
        from app.services.ollama_client import OllamaClient

        client: LLMClient = OllamaClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )

    elif provider == "openai_compatible":
        from app.services.openai_compatible_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    else:
        raise ValueError(
            f"不支持的 RAG_LLM_PROVIDER: {provider!r}，"
            "可选值：ollama / openai_compatible"
        )

    # 结构化类型检查：挡住「新适配器忘了实现 chat」这类实现漂移。
    # runtime_checkable 只校验方法是否存在，不校验签名，成本极低。
    if not isinstance(client, LLMClient):
        raise TypeError(
            f"{type(client).__name__} 未实现 LLMClient.chat()，"
            "请检查适配器是否漏了该方法"
        )

    logger.info(
        "LLM 客户端就绪：provider=%s model=%s base_url=%s",
        provider,
        settings.llm_model,
        settings.llm_base_url,
    )
    return client
