"""
协议无关的 LLM 抽象层。

定义统一消息、流式事件和 Provider 协议，让 TUI 不关心底层是 Anthropic 还是 OpenAI。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agentcode.config import ProviderConfig

ROLE_USER: Literal["user"] = "user"
ROLE_ASSISTANT: Literal["assistant"] = "assistant"
ROLE_TOOL: Literal["tool"] = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """协议无关的模型工具调用请求。"""

    id: str
    name: str
    input: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """协议无关的工具执行结果，用调用 id 与请求配对。"""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """协议无关的工具定义，由 provider 适配成各 SDK 格式。"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    # thinking 是独立通道，避免把模型推理内容混入最终可见回复。
    thinking: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    err: Exception | None = None


class Provider(Protocol):
    # 上层只依赖这个协议接口，因此新增后端时不需要改 TUI。
    @property
    def name(self) -> str:
        """返回用于状态栏展示的 provider 名称。"""
        ...

    @property
    def model(self) -> str:
        """返回当前 provider 配置的模型标识。"""
        ...

    def stream(
        self, msgs: list[Message], tools: list[ToolDefinition] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """把统一消息和工具定义转换为后端请求，并产出统一流事件。"""
        ...


def new_provider(cfg: ProviderConfig) -> Provider:
    """根据配置选择具体 Provider 适配器。"""

    # 适配器延迟导入，避免未选中的 SDK 在启动时产生额外副作用。
    if cfg.protocol == "anthropic":
        from agentcode.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from agentcode.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError(f"Unsupported protocol: {cfg.protocol}")
