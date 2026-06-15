"""
协议无关的 LLM 抽象层。

定义统一 Agent 消息、流式事件和 Provider 协议，让 Agent Core 不关心底层是
Anthropic 还是 OpenAI。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
import time
import traceback
from typing import Any, Literal, Protocol

from agentcode.config import ProviderConfig

ROLE_USER: Literal["user"] = "user"
ROLE_ASSISTANT: Literal["assistant"] = "assistant"
ROLE_TOOL_RESULT: Literal["toolResult"] = "toolResult"
KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
]
Api = str
KnownProvider = Literal[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "google",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "nvidia",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
]
ProviderName = str
ModelStopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
DoneStopReason = Literal["stop", "length", "toolUse"]
ErrorStopReason = Literal["error", "aborted"]
CacheRetention = Literal["none", "short", "long"]
Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
StreamHookResult = object | Awaitable[object | None] | None
PayloadHook = Callable[..., StreamHookResult]
ResponseHook = Callable[..., StreamHookResult]


def current_timestamp_ms() -> int:
    """返回统一消息协议使用的 Unix 毫秒时间戳。"""

    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class TextContent:
    """统一文本内容块，可出现在 user、assistant 和 toolResult 中。"""

    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


@dataclass(frozen=True, slots=True)
class ThinkingContent:
    """assistant thinking 内容块；Agent UI 可展示，但不混入最终可见文本。"""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass(frozen=True, slots=True)
class ImageContent:
    """统一图片内容块，当前协议层支持，内置工具暂时只产出文本。"""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = ""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """协议无关的模型工具调用请求，使用统一工具调用 block。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    type: Literal["toolCall"] = "toolCall"
    thought_signature: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """协议无关的工具定义，由 provider 适配成各 SDK 格式。"""

    name: str
    description: str
    parameters: dict[str, Any]
    prompt_snippet: str = ""
    prompt_guidelines: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UsageCost:
    """统一费用明细；当前没有计价表时保持为 0。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(frozen=True, slots=True)
class Usage:
    """统一 token 用量；provider 无法提供时字段保持为 0。"""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = field(default_factory=UsageCost)


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """StreamOptions.on_response 使用的 provider 响应摘要。"""

    status: int
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamOptions:
    """单次模型流请求的可选参数，承载统一 Agent 协议的请求选项。"""

    temperature: float | None = None
    max_tokens: int | None = None
    signal: object | None = None
    api_key: str | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    on_payload: PayloadHook | None = None
    on_response: ResponseHook | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int | None = None
    websocket_connect_timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


UserContent = str | list[TextContent | ImageContent]
AssistantContent = TextContent | ThinkingContent | ToolCall
ToolResultContent = TextContent | ImageContent


@dataclass(frozen=True, slots=True)
class DiagnosticErrorInfo:
    """统一诊断 error 形态，用于保存 provider 异常摘要。"""

    message: str
    name: str | None = None
    stack: str | None = None
    code: str | int | None = None


@dataclass(frozen=True, slots=True)
class AssistantMessageDiagnostic:
    """统一诊断结构，timestamp 使用 Unix 毫秒。"""

    type: str
    timestamp: int = field(default_factory=current_timestamp_ms)
    error: DiagnosticErrorInfo | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UserMessage:
    """用户输入消息，统一消息协议中的 user 消息。"""

    role: Literal["user"] = ROLE_USER
    content: UserContent = ""
    timestamp: int = field(default_factory=current_timestamp_ms)


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """模型回复消息，content 内保存 text/thinking/toolCall blocks。"""

    role: Literal["assistant"] = ROLE_ASSISTANT
    content: list[AssistantContent] = field(default_factory=list)
    api: Api = ""
    provider: ProviderName = ""
    model: str = ""
    response_model: str | None = None
    response_id: str | None = None
    diagnostics: list[AssistantMessageDiagnostic] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: ModelStopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=current_timestamp_ms)


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """单个工具调用的结果消息，统一消息协议中的 toolResult 消息。"""

    role: Literal["toolResult"] = ROLE_TOOL_RESULT
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultContent] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    timestamp: int = field(default_factory=current_timestamp_ms)


Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class Context:
    """模型请求上下文，集中携带 system prompt、历史消息和工具定义。"""

    messages: list[Message]
    system_prompt: str | None = None
    tools: list[ToolDefinition] | None = None


@dataclass(frozen=True, slots=True)
class StartEvent:
    """assistant 流开始事件，携带当前 partial message。"""

    type: Literal["start"] = "start"
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass(frozen=True, slots=True)
class TextStartEvent:
    """text block 开始事件，content_index 对应 assistant content 下标。"""

    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(frozen=True, slots=True)
class TextDeltaEvent:
    """text block 增量事件。"""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True)
class TextEndEvent:
    """text block 完成事件。"""

    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(frozen=True, slots=True)
class ThinkingStartEvent:
    """thinking block 开始事件。"""

    content_index: int
    partial: AssistantMessage
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(frozen=True, slots=True)
class ThinkingDeltaEvent:
    """thinking block 增量事件。"""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True, slots=True)
class ThinkingEndEvent:
    """thinking block 完成事件。"""

    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(frozen=True, slots=True)
class ToolCallStartEvent:
    """toolCall block 开始事件。"""

    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(frozen=True, slots=True)
class ToolCallDeltaEvent:
    """toolCall 参数增量事件，当前 provider 可不产生。"""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(frozen=True, slots=True)
class ToolCallEndEvent:
    """toolCall block 完成事件，携带结构化 ToolCall。"""

    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(frozen=True, slots=True)
class DoneEvent:
    """assistant 正常完成事件，reason 只能是统一协议的成功停止原因。"""

    reason: DoneStopReason
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """assistant 错误完成事件，error 内保存 stop_reason/error_message。"""

    reason: ErrorStopReason
    error: AssistantMessage
    err: Exception | None = None
    type: Literal["error"] = "error"


AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


class Provider(Protocol):
    # 上层只依赖这个协议接口，因此新增后端时不需要改 UI。
    @property
    def api(self) -> Api:
        """返回 provider 使用的 API 协议名。"""
        ...

    @property
    def name(self) -> ProviderName:
        """返回用于状态栏展示的 provider 名称。"""
        ...

    @property
    def model(self) -> str:
        """返回当前 provider 配置的模型标识。"""
        ...

    @property
    def context_window(self) -> int | None:
        """返回配置声明的上下文窗口；未知时由上下文管理模块使用协议默认值。"""
        ...

    def stream(
        self, context: Context, options: StreamOptions | None = None
    ) -> AsyncIterator[AssistantMessageEvent]:
        """把统一上下文转换为后端请求，并产出统一流事件。"""
        ...


def diagnostic_from_exception(
    kind: str,
    exc: BaseException,
    details: dict[str, Any] | None = None,
) -> AssistantMessageDiagnostic:
    """把 SDK 异常转换成统一诊断结构，避免 provider 各自拼字段。"""

    code = getattr(exc, "code", None)
    safe_code = code if isinstance(code, str | int) else None
    return AssistantMessageDiagnostic(
        type=kind,
        error=DiagnosticErrorInfo(
            message=str(exc) or type(exc).__name__,
            name=type(exc).__name__,
            stack="".join(traceback.format_exception(exc)),
            code=safe_code,
        ),
        details=details or {},
    )


def text_content(text: str) -> TextContent:
    """用最短路径创建文本内容块，避免调用处重复写 type 默认值。"""

    return TextContent(text=text)


def thinking_content(thinking: str, redacted: bool = False) -> ThinkingContent:
    """创建 thinking 内容块，redacted 用于安全过滤后的推理占位。"""

    return ThinkingContent(thinking=thinking, redacted=redacted)


def assistant_text(message: AssistantMessage) -> str:
    """提取 assistant content 中所有可见文本块。"""

    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


def assistant_tool_calls(message: AssistantMessage) -> list[ToolCall]:
    """从 assistant content 中按原始顺序取出所有工具调用块。"""

    return [block for block in message.content if isinstance(block, ToolCall)]


def message_text(message: Message) -> str:
    """提取 UI 和测试常用的可见文本，屏蔽不同 message content 形态。"""

    if isinstance(message, AssistantMessage):
        return assistant_text(message)
    if isinstance(message, UserMessage):
        return _content_to_text(message.content)
    return _content_to_text(message.content)


def tool_result_text(message: ToolResultMessage) -> str:
    """提取 toolResult 中所有文本块，provider 适配和 UI 展示共用。"""

    return _content_to_text(message.content)


def _content_to_text(content: UserContent | list[ToolResultContent]) -> str:
    """把字符串或文本 block 列表转换成纯文本，图片块不参与拼接。"""

    if isinstance(content, str):
        return content
    return "\n".join(
        block.text for block in content if isinstance(block, TextContent)
    )


def create_provider(cfg: ProviderConfig) -> Provider:
    """根据配置选择具体 Provider 适配器。"""

    # 适配器延迟导入，避免未选中的 SDK 在启动时产生额外副作用。
    if cfg.protocol == "anthropic":
        from agentcode.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from agentcode.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError(f"Unsupported protocol: {cfg.protocol}")
