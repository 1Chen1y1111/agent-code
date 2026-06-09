from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
import re

from agentcode.config import ProviderConfig
from agentcode.llm import Message, StreamEvent
from agentcode.tui import AgentCodeApp, SessionState
from agentcode.tui.input import ChatInput, _decode_csi_u_text
from agentcode.tui.scrollbar import PillScrollBarRender
from agentcode.tui.view import status_text
from textual.geometry import Offset
from textual.scrollbar import ScrollBar
from textual.widgets import OptionList, RichLog


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_provider_starts_idle(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            self.assertEqual(app.state, SessionState.IDLE)
            self.assertEqual(app.provider.name, "Only")

    async def test_multiple_providers_start_selecting(self) -> None:
        app = AgentCodeApp([_provider("One", "openai"), _provider("Two", "anthropic")])

        async with app.run_test():
            self.assertEqual(app.state, SessionState.SELECTING)
            self.assertIsNone(app.provider)

    async def test_layout_keeps_full_height_and_framed_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(40, 12)):
            screen_css = re.search(r"Screen\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            input_css = re.search(r"#input\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            log_css = re.search(r"#log\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            global_css = re.search(r"\*\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            ansi_css = re.search(
                r"App:ansi Screen,\s*App:ansi \*\s*\{(?P<body>.*?)\}",
                app.CSS,
                re.DOTALL,
            )

            self.assertIsNotNone(screen_css)
            self.assertIsNotNone(input_css)
            self.assertIsNotNone(log_css)
            self.assertIsNotNone(global_css)
            self.assertIsNotNone(ansi_css)
            self.assertIn("height: 100%;", screen_css.group("body"))
            self.assertIn("overflow-x: hidden;", screen_css.group("body"))
            self.assertIn("border: round $accent;", input_css.group("body"))
            self.assertIn("overflow-x: hidden;", log_css.group("body"))
            self.assertIn("overflow-y: auto;", log_css.group("body"))
            self.assertIn("scrollbar-size-vertical: 1;", global_css.group("body"))
            self.assertIn("scrollbar-size-horizontal: 0;", global_css.group("body"))
            self.assertIn(
                "scrollbar-background: transparent;", global_css.group("body")
            )
            self.assertIn("scrollbar-color: #a7a7b3;", global_css.group("body"))
            self.assertIn(
                "scrollbar-color: #a7a7b3 !important;", ansi_css.group("body")
            )

    async def test_global_scrollbars_are_vertical_only_and_low_profile(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(80, 24)):
            log = app.query_one("#log", RichLog)

            self.assertEqual(str(app.screen.styles.overflow_x), "hidden")
            self.assertEqual(app.screen.styles.scrollbar_size_vertical, 1)
            self.assertEqual(app.screen.styles.scrollbar_size_horizontal, 0)
            self.assertEqual(str(log.styles.overflow_x), "hidden")
            self.assertEqual(str(log.styles.overflow_y), "auto")
            self.assertEqual(log.styles.scrollbar_size_vertical, 1)
            self.assertEqual(log.styles.scrollbar_size_horizontal, 0)
            self.assertEqual(log.styles.scrollbar_background.hex, "#00000000")
            self.assertEqual(log.styles.scrollbar_color.hex, "#A7A7B3")
            self.assertIs(ScrollBar.renderer, PillScrollBarRender)

    async def test_provider_selector_uses_global_scrollbar_style(self) -> None:
        app = AgentCodeApp([_provider("One", "openai"), _provider("Two", "anthropic")])

        async with app.run_test(size=(80, 24)):
            selector = app.query_one("#selector", OptionList)

            self.assertEqual(str(selector.styles.overflow_x), "hidden")
            self.assertEqual(selector.styles.scrollbar_size_vertical, 1)
            self.assertEqual(selector.styles.scrollbar_size_horizontal, 0)
            self.assertEqual(selector.styles.scrollbar_background.hex, "#00000000")
            self.assertEqual(selector.styles.scrollbar_color.hex, "#A7A7B3")

    async def test_inline_layout_uses_full_terminal_height_before_log_scrolls(
        self,
    ) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(80, 24)):
            self.assertEqual(app._get_inline_height(), 24)  # noqa: SLF001

    def test_status_text_uses_lightweight_separator(self) -> None:
        self.assertEqual(
            status_text("Anthropic Claude", "deepseek-v4-pro").plain,
            "provider: Anthropic Claude · model: deepseek-v4-pro",
        )

    async def test_enter_submits_and_clears_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = FakeProvider("ok")
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "i", "enter")
            await pilot.pause(0.3)

            self.assertEqual(
                [(message.role, message.content) for message in app.conv.messages()],
                [("user", "hi"), ("assistant", "ok")],
            )
            self.assertEqual(input_box.text, "")

    async def test_alt_enter_inserts_newline_without_submit(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = FakeProvider("ok")
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "alt+enter", "i")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "h\ni")
            self.assertEqual(app.conv.messages(), [])

    async def test_chinese_input_keeps_multiple_characters(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("你", "好")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "你好")

    async def test_empty_focused_input_shows_cursor(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            self.assertIn(  # noqa: SLF001
                ChatInput.CURSOR_CHARACTER, input_box._render_input().plain
            )
            self.assertIn("Send a message...", input_box._render_input().plain)  # noqa: SLF001

    async def test_empty_focused_input_uses_codex_like_placeholder_layout(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            self.assertEqual(
                input_box._render_input().plain,  # noqa: SLF001
                f"❯ {ChatInput.CURSOR_CHARACTER}Send a message...",
            )

    async def test_click_focuses_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.blur()

            input_box.on_click()

            self.assertTrue(input_box.has_focus)
            self.assertIn(  # noqa: SLF001
                ChatInput.CURSOR_CHARACTER, input_box._render_input().plain
            )

    async def test_focused_cursor_blinks(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            self.assertIn(  # noqa: SLF001
                ChatInput.CURSOR_CHARACTER, input_box._render_input().plain
            )
            input_box._blink_cursor()  # noqa: SLF001
            self.assertNotIn(  # noqa: SLF001
                ChatInput.CURSOR_CHARACTER, input_box._render_input().plain
            )
            input_box._blink_cursor()  # noqa: SLF001
            self.assertIn(  # noqa: SLF001
                ChatInput.CURSOR_CHARACTER, input_box._render_input().plain
            )

    async def test_placeholder_does_not_shift_when_cursor_blinks(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            visible_text = input_box._render_input().plain  # noqa: SLF001
            input_box._blink_cursor()  # noqa: SLF001
            hidden_text = input_box._render_input().plain  # noqa: SLF001

            self.assertEqual(
                visible_text.index("Send a message..."),
                hidden_text.index("Send a message..."),
            )

    async def test_placeholder_does_not_shift_when_input_focuses(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.blur()
            blurred_text = input_box._render_input().plain  # noqa: SLF001

            input_box.focus()
            focused_text = input_box._render_input().plain  # noqa: SLF001

            self.assertEqual(
                blurred_text.index("Send a message..."),
                focused_text.index("Send a message..."),
            )

    async def test_disabled_input_hides_cursor(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            input_box.disabled = True

            self.assertNotIn(  # noqa: SLF001
                ChatInput.CURSOR_CHARACTER, input_box._render_input().plain
            )

    async def test_terminal_cursor_position_tracks_empty_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(80, 24)):
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2, input_box.content_region.y),
            )

    async def test_terminal_cursor_position_tracks_chinese_input_width(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(80, 24)):
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            input_box.insert("你好")

            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2 + 4, input_box.content_region.y),
            )

    async def test_terminal_cursor_position_tracks_multiline_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(80, 24)):
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            input_box.insert("a\n你好")

            self.assertEqual(
                app.cursor_position,
                Offset(
                    input_box.content_region.x + 2 + 4,
                    input_box.content_region.y + 1,
                ),
            )

    async def test_csi_u_ime_sequence_decodes_to_text(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)

            input_box._insert_printable("[32;;20320:21834:32418u")  # noqa: SLF001

            self.assertEqual(input_box.text, "你啊红")

    async def test_split_csi_u_ime_sequence_decodes_to_text(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)

            for character in "[32;;20320:21834:32418u":
                input_box._insert_printable(character)  # noqa: SLF001

            self.assertEqual(input_box.text, "你啊红")

    async def test_regular_bracket_input_is_preserved(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)

            input_box._insert_printable("[a")  # noqa: SLF001

            self.assertEqual(input_box.text, "[a")


class KeyboardSequenceTests(unittest.TestCase):
    def test_decode_csi_u_text(self) -> None:
        self.assertEqual(_decode_csi_u_text("[32;;20320:21834:32418u"), "你啊红")

    def test_decode_csi_u_text_with_escape_prefix(self) -> None:
        self.assertEqual(_decode_csi_u_text("\x1b[32;;20320:21834u"), "你啊")

    def test_decode_csi_u_text_rejects_non_sequence(self) -> None:
        self.assertIsNone(_decode_csi_u_text("[a"))


class ScrollBarRenderTests(unittest.TestCase):
    def test_pill_scrollbar_render_uses_capsule_glyphs(self) -> None:
        segments = PillScrollBarRender.render_bar(
            size=8,
            virtual_size=32,
            window_size=16,
            position=0,
            thickness=1,
            vertical=True,
        )
        text = "".join(segment.text for segment in segments.segments)
        track_segments = [
            segment for segment in segments.segments if segment.text == " "
        ]

        self.assertIn("▄", text)
        self.assertIn("█", text)
        self.assertIn("▀", text)
        self.assertNotIn("▗▖", text)
        self.assertNotIn("▝▘", text)
        self.assertTrue(track_segments)
        self.assertTrue(
            all(
                segment.style is None or segment.style.bgcolor is None
                for segment in track_segments
            )
        )


def _provider(name: str, protocol: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        protocol=protocol,  # type: ignore[arg-type]
        api_key="test-key",
        model="test-model",
    )


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.name = "Fake"
        self.model = "fake-model"
        self._reply = reply

    async def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(text=self._reply)
        yield StreamEvent(done=True)


if __name__ == "__main__":
    unittest.main()
