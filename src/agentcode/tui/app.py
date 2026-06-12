"""
AgentCode 的 Textual TUI 主应用。

负责 provider 选择、对话状态机、流式消费、计时展示和自绘聊天输入框。
"""

from __future__ import annotations

import asyncio
import math
import time
from enum import Enum
from pathlib import Path

from rich.console import RenderableType
from textual.containers import VerticalScroll
from textual.app import App, ComposeResult
from textual.color import Color
from textual.timer import Timer
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from agentcode import __version__
from agentcode.config import ProviderConfig
from agentcode.llm import ROLE_USER, Provider, create_provider, message_text
from agentcode.prompt import render_banner
from agentcode.session import AgentSession
from agentcode.tool import Registry, create_default_registry
from agentcode.tui.input import ChatInput
from agentcode.tui.scrollbar import install_scrollbar_renderer
from agentcode.tui.view import (
    assistant_final,
    assistant_streaming,
    elapsed_block,
    error_block,
    status_text,
    tool_done,
    tool_pending,
    user_block,
    working_text,
)

WORKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class SessionState(Enum):
    # TUI 只允许在 IDLE 状态提交；STREAMING 期间输入框禁用但界面仍可刷新。
    SELECTING = "selecting"
    IDLE = "idle"
    STREAMING = "streaming"


class MessageWidget(Static):
    """聊天消息组件，保留原始 Rich renderable 以便原地更新和测试观察。"""

    def __init__(self, renderable: RenderableType, classes: str) -> None:
        """保存初始 Rich renderable，便于后续原地刷新和测试读取。"""

        super().__init__(renderable, classes=classes)
        self.renderable = renderable

    def update_renderable(self, renderable: RenderableType) -> None:
        """同时更新缓存的 renderable 和 Textual widget 内容。"""

        self.renderable = renderable
        self.update(renderable)


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

    App:ansi #chat,
    App:ansi #selector {
        overflow-x: hidden !important;
    }

    #selector {
        height: 1fr;
        border: round $accent;
        padding: 1 2;
    }

    #chat {
        height: 1fr;
        min-height: 6;
        overflow-x: hidden;
        overflow-y: auto;
        padding: 1 0 1 2;
    }

    .message {
        width: 100%;
        height: auto;
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

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+t", "toggle_thinking", "Toggle thinking"),
    ]

    def __init__(
        self, providers: list[ProviderConfig], registry: Registry | None = None
    ) -> None:
        """初始化 TUI 状态，并注入 provider 配置和工具注册中心。"""

        super().__init__()
        install_scrollbar_renderer()
        self.providers = providers
        self._tool_registry = registry or create_default_registry()
        self.provider: Provider | None = None
        self.agent_session: AgentSession | None = None
        self._agent_session_provider: Provider | None = None
        self.state = SessionState.SELECTING if len(providers) > 1 else SessionState.IDLE
        self.cur_thinking = ""
        self.cur_reply = ""
        self.hide_thinking = False
        self._active_assistant: MessageWidget | None = None
        self._active_tool: MessageWidget | None = None
        self._working_widget: MessageWidget | None = None
        self._active_tool_name = ""
        self._active_tool_args = ""
        self.turn_start = 0.0
        self.turn_elapsed = 0
        self._working_frame_index = 0
        self._stream_task: asyncio.Task[None] | None = None
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """声明 Textual 组件树：provider 选择器、聊天区、输入框和状态栏。"""

        yield OptionList(
            *[
                Option(f"{provider.name} ({provider.model})")
                for provider in self.providers
            ],
            id="selector",
        )
        yield VerticalScroll(id="chat")
        yield ChatInput(
            "",
            id="input",
            placeholder="Send a message...",
        )
        yield Static("", id="statusbar")

    async def on_mount(self) -> None:
        """Textual 挂载完成后初始化 banner、provider 选择状态和焦点。"""

        # banner 作为第一条消息挂入聊天容器；后续 View 刷新不会重复渲染。
        await self._append_chat(
            render_banner(__version__, str(Path.cwd())),
            classes="message banner-message",
        )
        if len(self.providers) == 1:
            self.provider = create_provider(self.providers[0])
            self._ensure_agent_session()
            self.state = SessionState.IDLE
        self._sync_visibility()
        self._apply_scrollbar_theme()
        self._update_statusbar()
        if self.state is SessionState.IDLE:
            self._focus_input()
            self.call_after_refresh(self._focus_input)
        else:
            self.query_one("#selector", OptionList).focus()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """用户选中 provider 后创建会话并切换到聊天界面。"""

        self.provider = create_provider(self.providers[event.option_index])
        self._ensure_agent_session()
        self.state = SessionState.IDLE
        self._sync_visibility()
        self._apply_scrollbar_theme()
        self._update_statusbar()
        self._focus_input()
        self.call_after_refresh(self._focus_input)

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """接收自绘输入框的提交事件并交给统一提交入口。"""

        await self.submit(event.text)

    async def submit(self, text: str) -> None:
        """提交一条用户消息，启动本回合的 session 事件消费任务。"""

        message = text.strip()
        if not message or self.state is not SessionState.IDLE:
            return
        if message == "/exit":
            await self.action_quit()
            return
        if self.provider is None:
            return

        input_box = self.query_one("#input", ChatInput)
        if self._ensure_agent_session() is None:
            return
        input_box.add_history(message)
        input_box.load_text("")
        # 流式期间不接受新提交，但 Textual 事件循环仍继续运行。
        input_box.disabled = True

        self.cur_thinking = ""
        self.cur_reply = ""
        self._active_assistant = None
        self._active_tool = None
        self._active_tool_name = ""
        self._active_tool_args = ""
        self.turn_elapsed = 0
        self.turn_start = time.monotonic()
        self.state = SessionState.STREAMING
        self._working_frame_index = 0
        await self._show_working()
        # 计时器从请求发出前启动，覆盖“等待首 token”的时间。
        self._timer = self.set_interval(0.1, self._tick)
        self._stream_task = asyncio.create_task(self._consume_session_events(message))

    async def _consume_session_events(self, message: str) -> None:
        """消费 AgentSession 事件，并转换成聊天流中的可见 UI 更新。"""

        session = self._ensure_agent_session()
        if session is None:
            return
        try:
            async for event in session.prompt(message):
                if event.type == "error" and event.err is not None:
                    await self._finish_with_error(event.err)
                    return
                if (
                    event.type == "message_end"
                    and event.message is not None
                    and event.message.role == ROLE_USER
                ):
                    await self._append_chat(
                        user_block(message_text(event.message)),
                        classes="message user-message",
                    )
                if event.type == "message_update" and event.thinking:
                    self.cur_thinking += event.thinking
                    await self._update_assistant_live()
                if event.type == "message_update" and event.text:
                    self.cur_reply += event.text
                    await self._update_assistant_live()
                if event.type == "tool_execution_start":
                    # 工具调用会切开 assistant 回合：前言留在消息流里，
                    # 后续最终答复会创建新的 assistant live widget。
                    self.cur_thinking = ""
                    self.cur_reply = ""
                    self._active_assistant = None
                    self._active_tool_name = event.tool_name
                    self._active_tool_args = event.args
                    self._active_tool = await self._append_chat(
                        tool_pending(
                            event.tool_name,
                            event.args,
                            self._elapsed_seconds(),
                        ),
                        classes="message tool-message",
                    )
                if event.type == "tool_execution_end":
                    tool_widget = self._active_tool
                    if tool_widget is None:
                        tool_widget = await self._append_chat(
                            "",
                            classes="message tool-message",
                        )
                    tool_widget.update_renderable(
                        tool_done(
                            event.tool_name,
                            event.args,
                            event.result,
                            event.is_error,
                        )
                    )
                    self._active_tool = None
                    self._active_tool_name = ""
                    self._active_tool_args = ""
                    self._scroll_chat_to_end()
                if event.type == "agent_end":
                    await self._finish_with_assistant()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - UI 要恢复并展示用户可读错误。
            await self._finish_with_error(exc)

    def _tick(self) -> None:
        """定时刷新回合耗时、spinner 和正在执行的工具状态。"""

        if self.state is not SessionState.STREAMING:
            return
        self.turn_elapsed = int(time.monotonic() - self.turn_start)
        self._working_frame_index = (self._working_frame_index + 1) % len(
            WORKING_FRAMES
        )
        self._update_working()
        if self._active_tool is not None:
            self._active_tool.update_renderable(
                tool_pending(
                    self._active_tool_name,
                    self._active_tool_args,
                    self.turn_elapsed,
                )
            )
            self._scroll_chat_to_end()

    async def _finish_with_assistant(self) -> None:
        """在正常结束时补齐最终助手消息、耗时块并恢复输入状态。"""

        if self.cur_reply or self.cur_thinking:
            await self._update_assistant_final()
        await self._append_chat(
            elapsed_block(self._final_elapsed_seconds()),
            classes="message elapsed-message",
        )
        await self._finish_turn()

    async def _finish_with_error(self, error: Exception) -> None:
        """在流式请求或工具链路失败时显示错误并恢复输入状态。"""

        await self._append_chat(error_block(error), classes="message error-message")
        await self._finish_turn()

    async def _finish_turn(self) -> None:
        """清理当前回合的临时 UI 状态并重新允许用户输入。"""

        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._stream_task = None
        self.cur_thinking = ""
        self.cur_reply = ""
        self._active_assistant = None
        self._active_tool = None
        self._active_tool_name = ""
        self._active_tool_args = ""
        await self._hide_working()
        self.state = SessionState.IDLE
        input_box = self.query_one("#input", ChatInput)
        input_box.disabled = False
        self._focus_input()
        # Working 消失会改变聊天区最终高度；刷新后再滚一次，避免停在旧的最大滚动位置。
        self.call_after_refresh(self._scroll_chat_to_end)

    async def action_quit(self) -> None:
        """处理退出快捷键，并取消仍在运行的流式任务。"""

        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
        self.exit()

    def action_toggle_thinking(self) -> None:
        """切换当前流式助手消息中的 thinking 展示状态。"""

        self.hide_thinking = not self.hide_thinking
        if self._active_assistant is not None:
            self._active_assistant.update_renderable(self._assistant_renderable())
            self._scroll_chat_to_end()

    async def _append_chat(
        self, renderable: RenderableType, classes: str = "message"
    ) -> MessageWidget:
        """向聊天容器追加消息，并保持 working 临时消息始终位于末尾。"""

        restore_working = (
            self._working_widget is not None
            and "working-message" not in classes.split()
        )
        if restore_working:
            await self._remove_working()

        widget = MessageWidget(renderable, classes=classes)
        await self.query_one("#chat", VerticalScroll).mount(widget)
        if restore_working:
            await self._show_working()
        self._scroll_chat_to_end()
        return widget

    async def _update_assistant_live(self) -> None:
        """创建或刷新当前正在流式输出的助手消息。"""

        renderable = self._assistant_renderable()
        if self._active_assistant is None:
            self._active_assistant = await self._append_chat(
                renderable,
                classes="message assistant-message",
            )
            return
        self._active_assistant.update_renderable(renderable)
        self._scroll_chat_to_end()

    def _assistant_renderable(self) -> RenderableType:
        """根据当前 thinking/text 缓冲生成流式助手消息 renderable。"""

        return assistant_streaming(
            self.cur_thinking,
            self.cur_reply,
            self.hide_thinking,
        )

    async def _update_assistant_final(self) -> None:
        """把当前助手消息切换为最终 Markdown 渲染。"""

        renderable = assistant_final(
            self.cur_thinking,
            self.cur_reply,
            self.hide_thinking,
        )
        if self._active_assistant is None:
            self._active_assistant = await self._append_chat(
                renderable,
                classes="message assistant-message",
            )
            return
        self._active_assistant.update_renderable(renderable)
        self._scroll_chat_to_end()

    async def _show_working(self) -> None:
        """显示或刷新聊天流末尾的 Working 临时消息。"""

        renderable = working_text(WORKING_FRAMES[self._working_frame_index])
        if self._working_widget is None:
            # working 是聊天流的一条临时消息，而不是固定在输入框上方的状态栏；
            # 新消息追加时会先移除再挂回末尾，从而保持自然的滚动语义。
            self._working_widget = MessageWidget(
                renderable,
                classes="message working-message",
            )
            await self.query_one("#chat", VerticalScroll).mount(self._working_widget)
        else:
            self._working_widget.update_renderable(renderable)
        self._scroll_chat_to_end()

    def _update_working(self) -> None:
        """只刷新已存在的 Working 临时消息。"""

        if self._working_widget is None:
            return
        self._working_widget.update_renderable(
            working_text(WORKING_FRAMES[self._working_frame_index])
        )
        self._scroll_chat_to_end()

    async def _hide_working(self) -> None:
        """隐藏 Working 临时消息并保持滚动到底部。"""

        await self._remove_working()
        self._scroll_chat_to_end()

    async def _remove_working(self) -> None:
        """从聊天容器移除 Working 临时消息。"""

        if self._working_widget is None:
            return
        widget = self._working_widget
        self._working_widget = None
        await widget.remove()

    def _scroll_chat_to_end(self) -> None:
        """无动画滚动到聊天容器底部，匹配流式输出语义。"""

        self.query_one("#chat", VerticalScroll).scroll_end(
            animate=False,
            immediate=True,
            force=True,
        )

    def _focus_input(self) -> None:
        """把焦点放回自绘输入框。"""

        self.query_one("#input", ChatInput).focus()

    def _elapsed_seconds(self) -> int:
        """返回当前回合的实时整数耗时，用于运行中状态。"""

        return int(time.monotonic() - self.turn_start) if self.turn_start else 0

    def _final_elapsed_seconds(self) -> int:
        """返回最终展示耗时，短请求也至少显示 1 秒。"""

        if not self.turn_start:
            return 0
        return max(1, math.ceil(time.monotonic() - self.turn_start))

    def _ensure_agent_session(self) -> AgentSession | None:
        """确保当前 provider 对应的进程内 AgentSession 已创建。"""

        if self.provider is None:
            return None
        if (
            self.agent_session is None
            or self._agent_session_provider is not self.provider
        ):
            self.agent_session = AgentSession(self.provider, self._tool_registry)
            self._agent_session_provider = self.provider
        return self.agent_session

    def _sync_visibility(self) -> None:
        """根据当前状态切换 provider 选择界面和聊天界面的显示。"""

        selecting = self.state is SessionState.SELECTING
        self.query_one("#selector", OptionList).display = selecting
        self.query_one("#chat", VerticalScroll).display = not selecting
        self.query_one("#input", ChatInput).display = not selecting
        self.query_one("#statusbar", Static).display = not selecting

    def _update_statusbar(self) -> None:
        """把当前 provider 和 model 写入底部状态栏。"""

        statusbar = self.query_one("#statusbar", Static)
        if self.provider is None:
            statusbar.update("")
        else:
            statusbar.update(status_text(self.provider.name, self.provider.model))

    def _apply_scrollbar_theme(self) -> None:
        """运行时兜底设置滚动条主题，避免真实终端路径覆盖 CSS。"""

        track = Color.parse("transparent")
        thumb = Color.parse("#a7a7b3")
        thumb_hover = Color.parse("#b8b8c4")
        thumb_active = Color.parse("#cacad6")

        for widget in (
            self.screen,
            self.query_one("#selector", OptionList),
            self.query_one("#chat", VerticalScroll),
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
