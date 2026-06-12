from __future__ import annotations

import unittest

from agentcode.conversation import Conversation
from agentcode.llm import Message, ToolCall, ToolResult


class ConversationTests(unittest.TestCase):
    def test_adds_messages_in_order(self) -> None:
        conversation = Conversation()

        conversation.add_user("hello")
        conversation.add_assistant("hi")

        self.assertEqual(
            conversation.messages(),
            [
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
        )

    def test_messages_returns_copy(self) -> None:
        conversation = Conversation()
        conversation.add_user("hello")

        messages = conversation.messages()
        messages.append(Message(role="assistant", content="mutated"))

        self.assertEqual(
            conversation.messages(), [Message(role="user", content="hello")]
        )

    def test_adds_tool_messages_in_order(self) -> None:
        conversation = Conversation()
        call = ToolCall(id="call_1", name="read", input='{"path":"spec.md"}')
        result = ToolResult(
            tool_call_id="call_1", content="file content", is_error=False
        )

        conversation.add_user("read spec")
        conversation.add_assistant_with_tool_calls("I will read it.", [call])
        conversation.add_tool_results([result])
        conversation.add_assistant("done")

        self.assertEqual(
            conversation.messages(),
            [
                Message(role="user", content="read spec"),
                Message(
                    role="assistant",
                    content="I will read it.",
                    tool_calls=[call],
                ),
                Message(role="tool", tool_results=[result]),
                Message(role="assistant", content="done"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
