from __future__ import annotations

import unittest

from agentcode.conversation import Conversation
from agentcode.llm import Message


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


if __name__ == "__main__":
    unittest.main()
