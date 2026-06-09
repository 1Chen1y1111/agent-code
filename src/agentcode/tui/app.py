"""AgentCode 的 Textual TUI 主应用。

负责 provider 选择、对话状态机、流式消费、计时展示和自绘聊天输入框。
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.color import Color
from textual.timer import Timer
from textual.widgets import OptionList, RichLog, Static
from textual.widgets.option_list import Option

from agentcode import __version__
from agentcode.config import ProviderConfig
from agentcode.conversation import Conversation
from agentcode.llm import Provider, new_provider
from agentcode.prompt import render_banner
from agentcode.tui.input import ChatInput
from agentcode.tui.scrollbar import install_scrollbar_renderer
from agentcode.tui.view import (
    assistant_markdown,
    elapsed_block,
    error_block,
    status_text,
    user_block,
)


class SessionState(Enum):
    # TUI 只允许在 IDLE 状态提交；STREAMING 期间输入框禁用但界面仍可刷新。
    SELECTING = "selecting"
    IDLE = "idle"
    STREAMING = "streaming"


class AgentCodeApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        height: 100%;
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-color: #a7a7b3;
        scrollbar-color-hover: #b8b8c4;
        scrollbar-color-active: #cacad6;
        scrollbar-corner-color: transparent;
    }

    * {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-color: #a7a7b3;
        scrollbar-color-hover: #b8b8c4;
        scrollbar-color-active: #cacad6;
        scrollbar-corner-color: transparent;
    }

    .-ansi-scrollbar {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-color: #a7a7b3;
        scrollbar-color-hover: #b8b8c4;
        scrollbar-color-active: #cacad6;
        scrollbar-corner-color: transparent;
    }

    App:ansi Screen,
    App:ansi * {
        scrollbar-size-vertical: 1 !important;
        scrollbar-size-horizontal: 0 !important;
        scrollbar-background: transparent !important;
        scrollbar-background-hover: transparent !important;
        scrollbar-background-active: transparent !important;
        scrollbar-color: #a7a7b3 !important;
        scrollbar-color-hover: #b8b8c4 !important;
        scrollbar-color-active: #cacad6 !important;
        scrollbar-corner-color: transparent !important;
    }

    App:ansi #log,
    App:ansi #selector {
        overflow-x: hidden !important;
    }

    #selector {
        height: 1fr;
        border: round $accent;
        padding: 1 2;
    }

    #log {
        height: 1fr;
        min-height: 6;
        overflow-x: hidden;
        overflow-y: auto;
        padding: 1 0 1 2;
    }

    #streaming {
        padding: 1 0 1 2;
    }

    #input {
        height: auto;
        min-height: 1;
        border: round $accent;
        padding: 1;
    }

    #statusbar {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, providers: list[ProviderConfig]) -> None:
        super().__init__()
        install_scrollbar_renderer()
        self.providers = providers
        self.provider: Provider | None = None
        self.state = SessionState.SELECTING if len(providers) > 1 else SessionState.IDLE
        self.conv = Conversation()
        self.cur_reply = ""
        self.turn_start = 0.0
        self.turn_elapsed = 0
        self._stream_task: asyncio.Task[None] | None = None
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield OptionList(
            *[
                Option(f"{provider.name} ({provider.model})")
                for provider in self.providers
            ],
            id="selector",
        )
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Static("", id="streaming")
        yield ChatInput(
            "",
            id="input",
            placeholder="Send a message...",
        )
        yield Static("", id="statusbar")

    def on_mount(self) -> None:
        # banner 写入 RichLog 一次即可，后续 View 刷新不重复渲染。
        self.query_one("#log", RichLog).write(
            render_banner(__version__, str(Path.cwd()))
        )
        if len(self.providers) == 1:
            self.provider = new_provider(self.providers[0])
            self.state = SessionState.IDLE
        self._sync_visibility()
        self._apply_scrollbar_theme()
        self._update_statusbar()
        if self.state is SessionState.IDLE:
            self.query_one("#input", ChatInput).focus()
        else:
            self.query_one("#selector", OptionList).focus()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        self.provider = new_provider(self.providers[event.option_index])
        self.state = SessionState.IDLE
        self._sync_visibility()
        self._apply_scrollbar_theme()
        self._update_statusbar()
        self.query_one("#input", ChatInput).focus()

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        await self.submit(event.text)

    async def submit(self, text: str) -> None:
        message = text.strip()
        if not message or self.state is not SessionState.IDLE:
            return
        if message == "/exit":
            await self.action_quit()
            return
        if self.provider is None:
            return

        input_box = self.query_one("#input", ChatInput)
        log = self.query_one("#log", RichLog)
        self.conv.add_user(message)
        log.write(user_block(message))
        input_box.load_text("")
        # 流式期间不接受新提交，但 Textual 事件循环仍继续运行。
        input_box.disabled = True

        self.cur_reply = ""
        self.turn_elapsed = 0
        self.turn_start = time.monotonic()
        self.state = SessionState.STREAMING
        self._refresh_streaming_view()
        # 计时器从请求发出前启动，覆盖“等待首 token”的时间。
        self._timer = self.set_interval(0.2, self._tick)
        self._stream_task = asyncio.create_task(self._consume_stream())

    async def _consume_stream(self) -> None:
        if self.provider is None:
            return
        try:
            async for event in self.provider.stream(self.conv.messages()):
                if event.err is not None:
                    self._finish_with_error(event.err)
                    return
                if event.text:
                    # 流式期间直接显示纯文本，结束后再写入 Markdown 定型块。
                    self.cur_reply += event.text
                    self._refresh_streaming_view()
                if event.done:
                    self._finish_with_assistant()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - UI 要恢复并展示用户可读错误。
            self._finish_with_error(exc)

    def _tick(self) -> None:
        if self.state is not SessionState.STREAMING:
            return
        self.turn_elapsed = int(time.monotonic() - self.turn_start)
        self._refresh_streaming_view()

    def _finish_with_assistant(self) -> None:
        elapsed = int(time.monotonic() - self.turn_start)
        log = self.query_one("#log", RichLog)
        # RichLog 保留已完成消息；动态区只放当前正在生成的回复。
        log.write(assistant_markdown(self.cur_reply))
        log.write(elapsed_block(elapsed))
        self.conv.add_assistant(self.cur_reply)
        self._finish_turn()

    def _finish_with_error(self, error: Exception) -> None:
        log = self.query_one("#log", RichLog)
        log.write(error_block(error))
        self._finish_turn()

    def _finish_turn(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._stream_task = None
        self.cur_reply = ""
        self.state = SessionState.IDLE
        self.query_one("#streaming", Static).update("")
        input_box = self.query_one("#input", ChatInput)
        input_box.disabled = False
        input_box.focus()

    async def action_quit(self) -> None:
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
        self.exit()

    def _refresh_streaming_view(self) -> None:
        seconds = (
            int(time.monotonic() - self.turn_start)
            if self.turn_start
            else self.turn_elapsed
        )
        text = self.cur_reply or "Imagining..."
        self.query_one("#streaming", Static).update(
            Text(f"{text}\nImagining... ({seconds}s)")
        )

    def _sync_visibility(self) -> None:
        selecting = self.state is SessionState.SELECTING
        self.query_one("#selector", OptionList).display = selecting
        self.query_one("#log", RichLog).display = not selecting
        self.query_one("#streaming", Static).display = not selecting
        self.query_one("#input", ChatInput).display = not selecting
        self.query_one("#statusbar", Static).display = not selecting

    def _update_statusbar(self) -> None:
        statusbar = self.query_one("#statusbar", Static)
        if self.provider is None:
            statusbar.update("")
        else:
            statusbar.update(status_text(self.provider.name, self.provider.model))

    def _apply_scrollbar_theme(self) -> None:
        track = Color.parse("transparent")
        thumb = Color.parse("#a7a7b3")
        thumb_hover = Color.parse("#b8b8c4")
        thumb_active = Color.parse("#cacad6")

        for widget in (
            self.screen,
            self.query_one("#selector", OptionList),
            self.query_one("#log", RichLog),
        ):
            # ScrollBar.render() 读取父 widget 的 scrollbar_* 样式；这里做运行时兜底，
            # 避免 Textual ANSI 默认蓝色滚动条在真实终端路径覆盖 CSS。
            widget.styles.scrollbar_size_vertical = 1
            widget.styles.scrollbar_size_horizontal = 0
            widget.styles.scrollbar_background = track
            widget.styles.scrollbar_background_hover = track
            widget.styles.scrollbar_background_active = track
            widget.styles.scrollbar_color = thumb
            widget.styles.scrollbar_color_hover = thumb_hover
            widget.styles.scrollbar_color_active = thumb_active
            widget.styles.scrollbar_corner_color = track
