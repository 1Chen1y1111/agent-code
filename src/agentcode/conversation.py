"""单次进程会话的对话历史容器。

只保存 user/assistant 消息，用于每轮请求携带完整上下文，不负责持久化或压缩。
"""

from __future__ import annotations

from agentcode.llm import Message


class Conversation:
    # 会话历史只保存在进程内；退出后不落盘，也不做压缩或截断。
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_user(self, text: str) -> None:
        self._messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self._messages.append(Message(role="assistant", content=text))

    def messages(self) -> list[Message]:
        # 返回副本，避免 provider 或 UI 意外修改内部历史。
        return list(self._messages)
