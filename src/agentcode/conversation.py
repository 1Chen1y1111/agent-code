"""
单次进程会话的对话历史容器。

保存 user/assistant/tool 消息，用于每轮请求携带完整上下文，不负责持久化或压缩。
"""

from __future__ import annotations

from agentcode.llm import (
    AssistantContent,
    AssistantMessage,
    Message,
    ModelStopReason,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_content,
)


class Conversation:
    # 会话历史只保存在进程内；退出后不落盘，也不做压缩或截断。
    def __init__(self) -> None:
        """初始化当前进程内的一条线性对话历史。"""

        self._messages: list[Message] = []

    def add_user(self, text: str) -> None:
        """把用户输入追加为下一条 user 消息。"""

        self._messages.append(UserMessage(content=text))

    def add_assistant(
        self,
        text: str,
        usage: Usage | None = None,
        stop_reason: ModelStopReason | None = "stop",
    ) -> None:
        """把模型最终可见回复追加为 assistant 消息。"""

        self._messages.append(
            AssistantMessage(
                content=[text_content(text)] if text else [],
                usage=usage or Usage(),
                stop_reason=stop_reason or "stop",
            )
        )

    def add_assistant_message(self, message: AssistantMessage) -> None:
        """追加 provider 已构造好的完整 assistant 消息，保留元数据和 blocks。"""

        self._messages.append(message)

    def add_assistant_with_tool_calls(
        self,
        text: str,
        calls: list[ToolCall],
        usage: Usage | None = None,
        stop_reason: ModelStopReason | None = "toolUse",
    ) -> None:
        """保存包含工具调用请求的 assistant 消息，供 provider replay。"""

        content: list[AssistantContent] = []
        if text:
            content.append(text_content(text))
        content.extend(calls)
        self._messages.append(
            AssistantMessage(
                content=content,
                usage=usage or Usage(),
                stop_reason=stop_reason or "toolUse",
            )
        )

    def add_tool_results(self, results: list[ToolResultMessage]) -> None:
        """把工具执行结果逐条作为 toolResult 消息追加到历史。"""

        self._messages.extend(results)

    def messages(self) -> list[Message]:
        """返回历史副本，让调用方无法意外改动内部列表。"""

        # 返回副本，避免 provider 或 UI 意外修改内部历史。
        return list(self._messages)
