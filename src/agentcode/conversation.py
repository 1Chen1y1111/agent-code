"""
单次进程会话的对话历史容器。

保存模型可见 active 历史和完整 archive 历史；不负责落盘、恢复或 provider 调用。
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
    # 会话历史只保存在进程内；archive 保留原始事实，active 可被压缩摘要替换。
    def __init__(self) -> None:
        """初始化当前进程内的一条线性对话历史。"""

        self._messages: list[Message] = []
        self._archive: list[Message] = []

    def add_user(self, text: str) -> None:
        """把用户输入追加为下一条 user 消息。"""

        message = UserMessage(content=text)
        self._messages.append(message)
        self._archive.append(message)

    def add_assistant(
        self,
        text: str,
        usage: Usage | None = None,
        stop_reason: ModelStopReason | None = "stop",
    ) -> None:
        """把模型最终可见回复追加为 assistant 消息。"""

        message = AssistantMessage(
            content=[text_content(text)] if text else [],
            usage=usage or Usage(),
            stop_reason=stop_reason or "stop",
        )
        self._messages.append(message)
        self._archive.append(message)

    def add_assistant_message(self, message: AssistantMessage) -> None:
        """追加 provider 已构造好的完整 assistant 消息，保留元数据和 blocks。"""

        self._messages.append(message)
        self._archive.append(message)

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
        message = AssistantMessage(
            content=content,
            usage=usage or Usage(),
            stop_reason=stop_reason or "toolUse",
        )
        self._messages.append(message)
        self._archive.append(message)

    def add_tool_results(
        self,
        results: list[ToolResultMessage],
        archive_results: list[ToolResultMessage] | None = None,
    ) -> None:
        """把工具结果追加到 active 历史，同时 archive 可保留原始结果。"""

        self._messages.extend(results)
        self._archive.extend(archive_results or results)

    def replace_active(self, messages: list[Message]) -> None:
        """用压缩后的消息替换模型可见历史，archive 继续保留原始事实。"""

        self._messages = list(messages)
        self._archive.extend(
            message for message in messages if message not in self._archive
        )

    def messages(self) -> list[Message]:
        """返回模型可见 active 历史副本。"""

        # 返回副本，避免 provider 或 UI 意外修改内部历史。
        return list(self._messages)

    def archive_messages(self) -> list[Message]:
        """返回完整 archive 历史副本，供调试或后续持久化使用。"""

        return list(self._archive)
