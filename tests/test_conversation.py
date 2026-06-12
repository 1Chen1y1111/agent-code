from __future__ import annotations

import unittest

from agentcode.conversation import Conversation
from agentcode.llm import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    assistant_tool_calls,
    message_text,
    text_content,
)


class ConversationTests(unittest.TestCase):
    def test_adds_messages_in_order(self) -> None:
        conversation = Conversation()

        conversation.add_user("hello")
        conversation.add_assistant("hi")

        messages = conversation.messages()
        self.assertEqual(
            [(message.role, message_text(message)) for message in messages],
            [("user", "hello"), ("assistant", "hi")],
        )
        self.assertIsInstance(messages[0].timestamp, int)
        self.assertEqual(messages[1].stop_reason, "stop")

    def test_messages_returns_copy(self) -> None:
        conversation = Conversation()
        conversation.add_user("hello")

        messages = conversation.messages()
        messages.append(AssistantMessage(content=[text_content("mutated")]))

        self.assertEqual(
            [
                (message.role, message_text(message))
                for message in conversation.messages()
            ],
            [("user", "hello")],
        )

    def test_adds_tool_messages_in_order(self) -> None:
        conversation = Conversation()
        call = ToolCall(id="call_1", name="read", arguments={"path": "spec.md"})
        result = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="read",
            content=[TextContent(text="file content")],
            is_error=False,
        )

        conversation.add_user("read spec")
        conversation.add_assistant_with_tool_calls("I will read it.", [call])
        conversation.add_tool_results([result])
        conversation.add_assistant("done")

        messages = conversation.messages()
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(message_text(messages[0]), "read spec")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(message_text(messages[1]), "I will read it.")
        self.assertEqual(messages[2].role, "toolResult")
        self.assertEqual(messages[2].content, [TextContent(text="file content")])
        self.assertEqual(messages[3].role, "assistant")
        self.assertEqual(message_text(messages[3]), "done")
        self.assertEqual(assistant_tool_calls(messages[1]), [call])
        self.assertEqual(messages[1].stop_reason, "toolUse")
        self.assertEqual(messages[2].tool_call_id, "call_1")
        self.assertEqual(messages[2].tool_name, "read")
        self.assertFalse(messages[2].is_error)


if __name__ == "__main__":
    unittest.main()
