from __future__ import annotations

import asyncio
import io
import re
import unittest
from collections.abc import AsyncIterator

from rich.console import Console

from agentcode.config import ProviderConfig
from agentcode.llm import Message, StreamEvent, ToolCall
from agentcode.tool import Registry, Result
from agentcode.tui import AgentCodeApp, SessionState
from agentcode.tui.input import ChatInput, _decode_csi_u_text
from agentcode.tui.scrollbar import PillScrollBarRender
from agentcode.tui.view import status_text, working_text
from textual.containers import VerticalScroll
from textual.geometry import Offset
from textual.scrollbar import ScrollBar
from textual.widgets import OptionList, Static


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

    async def test_selecting_provider_focuses_input(self) -> None:
        app = AgentCodeApp([_provider("One", "openai"), _provider("Two", "anthropic")])

        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause(0.2)

            input_box = app.query_one("#input", ChatInput)
            self.assertEqual(app.state, SessionState.IDLE)
            self.assertTrue(input_box.has_focus)

            await pilot.press("h", "i")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "hi")

    async def test_layout_keeps_full_height_and_framed_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test(size=(40, 12)):
            screen_css = re.search(r"Screen\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            input_css = re.search(r"#input\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            chat_css = re.search(r"#chat\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            global_css = re.search(r"\*\s*\{(?P<body>.*?)\}", app.CSS, re.DOTALL)
            ansi_css = re.search(
                r"App:ansi Screen,\s*App:ansi \*\s*\{(?P<body>.*?)\}",
                app.CSS,
                re.DOTALL,
            )

            self.assertIsNotNone(screen_css)
            self.assertIsNotNone(input_css)
            self.assertIsNotNone(chat_css)
            self.assertIsNotNone(global_css)
            self.assertIsNotNone(ansi_css)
            self.assertIn("height: 100%;", screen_css.group("body"))
            self.assertIn("overflow-x: hidden;", screen_css.group("body"))
            self.assertIn("border: round $accent;", input_css.group("body"))
            self.assertIn("overflow-x: hidden;", chat_css.group("body"))
            self.assertIn("overflow-y: auto;", chat_css.group("body"))
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
            chat = app.query_one("#chat", VerticalScroll)

            self.assertEqual(str(app.screen.styles.overflow_x), "hidden")
            self.assertEqual(app.screen.styles.scrollbar_size_vertical, 1)
            self.assertEqual(app.screen.styles.scrollbar_size_horizontal, 0)
            self.assertEqual(str(chat.styles.overflow_x), "hidden")
            self.assertEqual(str(chat.styles.overflow_y), "auto")
            self.assertEqual(chat.styles.scrollbar_size_vertical, 1)
            self.assertEqual(chat.styles.scrollbar_size_horizontal, 0)
            self.assertEqual(chat.styles.scrollbar_background.hex, "#00000000")
            self.assertEqual(chat.styles.scrollbar_color.hex, "#A7A7B3")
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

    def test_working_text_uses_spinner_and_message(self) -> None:
        self.assertEqual(working_text("⠋").plain, "⠋ Working...")

    async def test_working_indicator_uses_chat_flow_not_fixed_row(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            ids = [
                child.id
                for child in app.screen.children
                if child.id in {"chat", "input", "statusbar"}
            ]

            self.assertEqual(ids, ["chat", "input", "statusbar"])

    async def test_enter_submits_and_clears_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = FakeProvider("ok")
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "i", "enter")
            await pilot.pause(0.3)

            self.assertEqual(
                [(message.role, message.content) for message in _messages(app)],
                [("user", "hi"), ("assistant", "ok")],
            )
            self.assertEqual(input_box.text, "")
            self.assertIsInstance(
                app.query_one("#chat", VerticalScroll), VerticalScroll
            )

            await pilot.press("up")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "hi")

    async def test_tool_events_render_inline_in_chat(self) -> None:
        registry = Registry()
        registry.register(FakeTool("read", "file content"))
        app = AgentCodeApp([_provider("Only", "openai")], registry=registry)

        async with app.run_test() as pilot:
            app.provider = ScriptedProvider(
                [
                    [
                        StreamEvent(
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="read",
                                    input='{"path":"note.txt"}',
                                )
                            ]
                        ),
                        StreamEvent(done=True),
                    ],
                    [StreamEvent(text="done"), StreamEvent(done=True)],
                ]
            )
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("r", "e", "a", "d", "enter")
            await pilot.pause(0.3)

            chat_text = _chat_text(app)
            self.assertIn("● read", chat_text)
            self.assertIn('read({"path":"note.txt"})', chat_text)
            self.assertIn("file content", chat_text)
            self.assertIn("done", chat_text)

    async def test_thinking_stream_renders_before_visible_answer(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = ScriptedProvider(
                [
                    [
                        StreamEvent(thinking="先分析边界"),
                        StreamEvent(text="最终答案"),
                        StreamEvent(done=True),
                    ]
                ]
            )
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "i", "enter")
            await pilot.pause(0.3)

            chat_text = _chat_text(app)
            self.assertIn("先分析边界", chat_text)
            self.assertIn("最终答案", chat_text)
            self.assertEqual(_messages(app)[-1].content, "最终答案")

    async def test_ctrl_t_hides_streaming_thinking_block(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            provider = ControlledThinkingProvider("先分析边界", "最终答案")
            app.provider = provider
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "i", "enter")
            for _ in range(20):
                if "先分析边界" in _chat_text(app):
                    break
                await pilot.pause(0.05)
            else:
                self.fail("thinking block did not render")

            await pilot.press("ctrl+t")
            await pilot.pause(0.05)

            chat_text = _chat_text(app)
            self.assertIn("Thinking...", chat_text)
            self.assertNotIn("先分析边界", chat_text)

            provider.resume()
            for _ in range(20):
                if "最终答案" in _chat_text(app):
                    break
                await pilot.pause(0.05)
            else:
                self.fail("final answer did not render")

            chat_text = _chat_text(app)
            self.assertIn("Thinking...", chat_text)
            self.assertIn("最终答案", chat_text)

    async def test_working_indicator_shows_animates_and_hides(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = SlowThinkingProvider("先分析边界", "最终答案")
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "i", "enter")
            await pilot.pause(0.03)

            first_frame = _working_text(app)
            self.assertEqual(len(_working_widgets(app)), 1)
            self.assertIn("Working...", first_frame)
            self.assertIn("Working...", _chat_text(app))

            await pilot.pause(0.13)

            self.assertNotEqual(_working_text(app), first_frame)

            await pilot.pause(0.3)

            self.assertEqual(_working_widgets(app), [])
            self.assertNotIn("Working...", _chat_text(app))

    async def test_working_indicator_hides_after_stream_error(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = ScriptedProvider([[StreamEvent(err=RuntimeError("boom"))]])
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "i", "enter")
            await pilot.pause(0.2)

            self.assertEqual(_working_widgets(app), [])
            self.assertNotIn("Working...", _chat_text(app))
            self.assertIn("请求失败：boom", _chat_text(app))

    async def test_working_indicator_stays_visible_during_tool_execution(self) -> None:
        registry = Registry()
        tool = BlockingTool("read", "file content")
        registry.register(tool)
        app = AgentCodeApp([_provider("Only", "openai")], registry=registry)

        async with app.run_test() as pilot:
            app.provider = ScriptedProvider(
                [
                    [
                        StreamEvent(
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="read",
                                    input='{"path":"note.txt"}',
                                )
                            ]
                        ),
                        StreamEvent(done=True),
                    ],
                    [StreamEvent(text="done"), StreamEvent(done=True)],
                ]
            )
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("r", "e", "a", "d", "enter")
            for _ in range(40):
                chat_text = _chat_text(app)
                if _working_widgets(app) and "Running..." in chat_text:
                    break
                await pilot.pause(0.05)
            else:
                self.fail(
                    "working indicator did not appear during tool execution:\n"
                    + _chat_text(app)
                )

            chat_text = _chat_text(app)
            self.assertIn("Working...", chat_text)
            self.assertIn("Running...", chat_text)
            tool.resume()

            for _ in range(40):
                if not _working_widgets(app) and "done" in _chat_text(app):
                    break
                await pilot.pause(0.05)
            else:
                self.fail("working indicator did not hide after tool execution")

            self.assertEqual(_working_widgets(app), [])

    async def test_alt_enter_inserts_newline_without_submit(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            app.provider = FakeProvider("ok")
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("h", "alt+enter", "i")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "h\ni")
            self.assertEqual(_messages(app), [])

    async def test_cursor_inserts_text_in_middle(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("a", "c", "left", "b")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "abc")
            self.assertEqual(input_box._render_input().plain, "❯ abc")  # noqa: SLF001

            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2 + 2, input_box.content_region.y),
            )

    async def test_non_empty_hardware_cursor_does_not_shift_text(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()
            input_box.load_text("abc")
            input_box._move_cursor(1)  # noqa: SLF001

            self.assertEqual(input_box._render_input().plain, "❯ abc")  # noqa: SLF001
            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2 + 1, input_box.content_region.y),
            )

    async def test_delete_keys_edit_around_cursor(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("a", "b", "c", "left", "left", "delete")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "ac")

            await pilot.press("backspace")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "c")

    async def test_home_and_end_move_within_current_line(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            await pilot.press("a", "alt+enter", "b", "c", "home", "x", "end", "y")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "a\nxbcy")

    async def test_history_navigation_preserves_current_draft(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test() as pilot:
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()
            input_box.add_history("one")
            input_box.add_history("two")

            await pilot.press("d", "r", "a", "f", "t", "up")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "two")

            await pilot.press("up")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "one")

            await pilot.press("down")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "two")

            await pilot.press("down")
            await pilot.pause(0.1)

            self.assertEqual(input_box.text, "draft")

    async def test_large_paste_collapses_but_submits_original_text(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            pasted = "\n".join(f"line {index}" for index in range(11))

            input_box._handle_paste(pasted)  # noqa: SLF001

            self.assertEqual(input_box.text, "[paste #1 +11 lines]")
            self.assertEqual(input_box.submitted_text(), pasted)

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

            self.assertEqual(  # noqa: SLF001
                input_box._render_input().plain,
                "❯ Send a message...",
            )
            self.assertIn("Send a message...", input_box._render_input().plain)  # noqa: SLF001
            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2, input_box.content_region.y),
            )

    async def test_empty_focused_input_uses_codex_like_placeholder_layout(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            self.assertEqual(
                input_box._render_input().plain,  # noqa: SLF001
                "❯ Send a message...",
            )

    async def test_click_focuses_input(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.blur()

            input_box.on_click()

            self.assertTrue(input_box.has_focus)
            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2, input_box.content_region.y),
            )

    async def test_focused_input_uses_hardware_cursor(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            self.assertEqual(
                app.cursor_position,
                Offset(input_box.content_region.x + 2, input_box.content_region.y),
            )

    async def test_input_focus_toggles_terminal_cursor_visibility(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            driver = CursorDriver()
            original_driver = app._driver  # noqa: SLF001
            app._driver = driver  # noqa: SLF001
            try:
                input_box._set_terminal_cursor_visible(True)  # noqa: SLF001
                input_box._set_terminal_cursor_visible(False)  # noqa: SLF001
            finally:
                app._driver = original_driver  # noqa: SLF001

            self.assertEqual(driver.writes, ["\x1b[?25h", "\x1b[?25l"])
            self.assertEqual(driver.flushes, 2)

    async def test_placeholder_does_not_shift_when_input_refreshes(self) -> None:
        app = AgentCodeApp([_provider("Only", "openai")])

        async with app.run_test():
            input_box = app.query_one("#input", ChatInput)
            input_box.focus()

            visible_text = input_box._render_input().plain  # noqa: SLF001
            input_box._refresh_display()  # noqa: SLF001
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

            self.assertNotIn("│", input_box._render_input().plain)  # noqa: SLF001

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

    async def stream(
        self, msgs: list[Message], tools: list[object] | None = None
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(text=self._reply)
        yield StreamEvent(done=True)


class ScriptedProvider:
    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self.name = "Fake"
        self.model = "fake-model"
        self._scripts = scripts

    async def stream(
        self, msgs: list[Message], tools: list[object] | None = None
    ) -> AsyncIterator[StreamEvent]:
        script = self._scripts.pop(0)
        for event in script:
            yield event


class SlowThinkingProvider:
    def __init__(self, thinking: str, reply: str, delay: float = 0.2) -> None:
        self.name = "Fake"
        self.model = "fake-model"
        self._thinking = thinking
        self._reply = reply
        self._delay = delay

    async def stream(
        self, msgs: list[Message], tools: list[object] | None = None
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(thinking=self._thinking)
        await asyncio.sleep(self._delay)
        yield StreamEvent(text=self._reply)
        yield StreamEvent(done=True)


class ControlledThinkingProvider:
    def __init__(self, thinking: str, reply: str) -> None:
        self.name = "Fake"
        self.model = "fake-model"
        self._thinking = thinking
        self._reply = reply
        self._resume = asyncio.Event()

    def resume(self) -> None:
        self._resume.set()

    async def stream(
        self, msgs: list[Message], tools: list[object] | None = None
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(thinking=self._thinking)
        await self._resume.wait()
        yield StreamEvent(text=self._reply)
        yield StreamEvent(done=True)


class FakeTool:
    def __init__(self, name: str, content: str) -> None:
        self._name = name
        self._content = content

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "fake"

    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    async def execute(self, args: str) -> Result:
        return Result(self._content)


class BlockingTool(FakeTool):
    def __init__(self, name: str, content: str) -> None:
        super().__init__(name, content)
        self._resume = asyncio.Event()

    def resume(self) -> None:
        self._resume.set()

    async def execute(self, args: str) -> Result:
        await self._resume.wait()
        return Result(self._content)


class CursorDriver:
    is_headless = False

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushes = 0

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        self.flushes += 1


def _chat_text(app: AgentCodeApp) -> str:
    console = Console(width=120, record=True, file=io.StringIO())
    for widget in app.query_one("#chat", VerticalScroll).query(Static):
        console.print(getattr(widget, "renderable", widget.render()))
    return console.export_text()


def _messages(app: AgentCodeApp) -> list[Message]:
    assert app.agent_session is not None
    return app.agent_session.messages()


def _working_widgets(app: AgentCodeApp) -> list[Static]:
    return [
        widget
        for widget in app.query_one("#chat", VerticalScroll).query(Static)
        if "working-message" in widget.classes
    ]


def _working_text(app: AgentCodeApp) -> str:
    console = Console(width=120, record=True, file=io.StringIO())
    for widget in _working_widgets(app):
        console.print(getattr(widget, "renderable", widget.render()))
    return console.export_text()


def _static_text(widget: Static) -> str:
    console = Console(width=120, record=True, file=io.StringIO())
    console.print(widget.render())
    return console.export_text()


if __name__ == "__main__":
    unittest.main()
