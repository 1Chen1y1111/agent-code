"""AgentCode 的普通终端 CLI 应用。

负责 provider 选择、prompt_toolkit 输入循环，以及把 Session 事件追加渲染到终端 scrollback。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from pathlib import Path
import time
from typing import Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.status import Status

from agentcode import __version__
from agentcode.config import ProviderConfig
from agentcode.llm import Provider, create_provider, message_text
from agentcode.prompt import PromptBuildOptions, render_banner
from agentcode.session import AgentSession, SessionEvent
from agentcode.tool import Registry, create_default_registry
from agentcode.terminal.view import (
    assistant_markdown,
    elapsed_block,
    error_block,
    provider_option,
    thinking_delta,
    tool_result_summary,
    tool_start,
    tool_update,
    user_block,
)

EXIT_COMMANDS = {"/exit", "/quit"}
PROMPT_TEXT = "❯ "
PROMPT_STYLE = Style.from_dict(
    {
        "bottom-toolbar": "noreverse bg:default ansibrightblack",
    }
)


class PromptReader(Protocol):
    """抽象 prompt_toolkit 输入会话，方便测试注入假输入。"""

    async def prompt_async(self, message: str, **kwargs: object) -> str:
        """异步读取一条用户输入；EOF/KeyboardInterrupt 由实现抛出。"""
        ...


class TerminalApp:
    """普通终端交互应用，使用终端自身 scrollback 承载历史和复制。"""

    def __init__(
        self,
        providers: list[ProviderConfig],
        registry: Registry | None = None,
        prompt_options: PromptBuildOptions | None = None,
        *,
        console: Console | None = None,
        prompt_reader: PromptReader | None = None,
        provider_factory: Callable[[ProviderConfig], Provider] = create_provider,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """保存启动依赖；真实 provider 和 session 会在用户选定后创建。"""

        self.providers = providers
        self._registry = registry or create_default_registry()
        self._prompt_options = prompt_options or PromptBuildOptions()
        self._console = console or Console()
        self._prompt_reader = prompt_reader or PromptSession(
            history=InMemoryHistory(),
            erase_when_done=True,
        )
        self._provider_factory = provider_factory
        self._clock = clock
        self.provider: Provider | None = None
        self.agent_session: AgentSession | None = None
        self._renderer = TerminalRenderer(self._console, self._clock)

    def run(self) -> None:
        """启动同步 CLI 入口，并把 Ctrl+C 作为普通退出处理。"""

        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            self._renderer.stop_status()

    async def run_async(self) -> None:
        """执行 provider 选择和主输入循环，直到用户退出或输入流结束。"""

        self._console.print(render_banner(__version__, str(Path.cwd())), end="")
        provider_config = await self._select_provider_config()
        if provider_config is None:
            return

        self.provider = self._provider_factory(provider_config)
        self.agent_session = AgentSession(
            self.provider,
            self._registry,
            self._prompt_options,
        )

        while True:
            try:
                raw_text = await self._prompt_reader.prompt_async(
                    PROMPT_TEXT,
                    bottom_toolbar=self._bottom_toolbar(),
                    style=PROMPT_STYLE,
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            message = raw_text.strip()
            if not message:
                continue
            if message in EXIT_COMMANDS:
                break
            await self._run_turn(message)

    async def _select_provider_config(self) -> ProviderConfig | None:
        """根据配置数量选择 provider；多 provider 时循环读取编号。"""

        if not self.providers:
            return None
        if len(self.providers) == 1:
            return self.providers[0]

        self._console.print("请选择 provider：", style="bold")
        self._console.print("")
        for index, provider in enumerate(self.providers, start=1):
            self._console.print(provider_option(index, provider.name, provider.model))
            self._console.print("")

        while True:
            try:
                selection = await self._prompt_reader.prompt_async("provider> ")
            except EOFError:
                return None
            except KeyboardInterrupt:
                return None

            provider = self._provider_from_selection(selection)
            if provider is not None:
                return provider
            self._console.print("请输入有效的 provider 编号。", style="bold red")

    def _provider_from_selection(self, selection: str) -> ProviderConfig | None:
        """把用户输入的编号转换成 provider 配置；非法输入返回 None。"""

        try:
            index = int(selection.strip())
        except ValueError:
            return None
        if 1 <= index <= len(self.providers):
            return self.providers[index - 1]
        return None

    def _bottom_toolbar(self) -> str:
        """生成 prompt_toolkit 底部状态栏文本，显示当前 provider 和 model。"""

        if self.provider is None:
            return ""
        return f"\nprovider: {self.provider.name} · model: {self.provider.model}"

    async def _run_turn(self, message: str) -> None:
        """提交一条用户消息并把 Session 事件流渲染到终端。"""

        if self.agent_session is None:
            return

        try:
            async for event in self.agent_session.prompt(message):
                self._renderer.render(event)
        except asyncio.CancelledError:
            self._renderer.stop_status()
            raise
        except Exception as exc:  # noqa: BLE001 - UI 层要恢复终端并展示错误。
            self._renderer.stop_status()
            self._console.print(error_block(exc), end="")
        finally:
            self._renderer.stop_status()


class TerminalRenderer:
    """把 SessionEvent 转换成普通终端的追加式输出。"""

    def __init__(
        self,
        console: Console,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """初始化输出状态；状态只影响当前回合的换行和 spinner。"""

        self._console = console
        self._clock = clock
        self._turn_start = 0.0
        self._status: Status | None = None
        self._assistant_has_output = False
        self._assistant_has_text_output = False
        self._assistant_last_char = ""
        self._assistant_last_kind = ""
        self._assistant_text_chunks: list[str] = []
        self._pending_block_gap = False
        self._rendered_tool_titles: set[str] = set()

    def render(self, event: SessionEvent) -> None:
        """分发并渲染单个 SessionEvent。"""

        if event.type == "agent_start":
            self._start_turn()
            return
        if event.type == "message_start":
            self._handle_message_start(event)
            return
        if event.type == "message_update":
            self._handle_message_update(event)
            return
        if event.type == "message_end":
            self._handle_message_end(event)
            return
        if event.type == "tool_execution_start":
            self._handle_tool_start(event)
            return
        if event.type == "tool_execution_update":
            self._handle_tool_update(event)
            return
        if event.type == "tool_execution_end":
            self._handle_tool_end(event)
            return
        if event.type == "error" and event.err is not None:
            self.stop_status()
            self._ensure_newline()
            self._console.print(error_block(event.err), end="")
            return
        if event.type == "agent_end":
            self._finish_turn()

    def stop_status(self) -> None:
        """停止临时状态，确保 Working 或工具运行提示不会留在 scrollback 中。"""

        if self._status is None:
            return
        self._status.__exit__(None, None, None)
        self._status = None

    def _start_turn(self) -> None:
        """记录回合开始时间并重置当前 assistant 输出状态。"""

        self._turn_start = self._clock()
        self._assistant_has_output = False
        self._assistant_has_text_output = False
        self._assistant_last_char = ""
        self._assistant_last_kind = ""
        self._assistant_text_chunks.clear()
        self._pending_block_gap = False
        self._rendered_tool_titles.clear()

    def _handle_message_start(self, event: SessionEvent) -> None:
        """在 assistant 消息开始时重置本段输出状态。"""

        if event.message is not None and event.message.role == "assistant":
            self._assistant_has_output = False
            self._assistant_has_text_output = False
            self._assistant_last_char = ""
            self._assistant_last_kind = ""
            self._assistant_text_chunks.clear()

    def _handle_message_update(self, event: SessionEvent) -> None:
        """实时追加 thinking，并缓存正文等待 Markdown 完整渲染。"""

        if event.thinking:
            self._stop_status_for_visible_output()
            self._ensure_pending_block_gap()
            self._console.print(thinking_delta(event.thinking), end="")
            self._mark_assistant_output(event.thinking, "thinking")
        if event.text:
            self._assistant_text_chunks.append(event.text)

    def _handle_message_end(self, event: SessionEvent) -> None:
        """在 user 或 assistant 消息结束时补齐对应的终端输出。"""

        if event.message is None:
            return
        if event.message.role == "user":
            self._console.print(user_block(message_text(event.message)), end="")
            self._pending_block_gap = True
            self._ensure_pending_block_gap()
            self._start_status()
            return
        if event.message.role == "assistant":
            self._render_assistant_text(message_text(event.message))
            self._ensure_newline()

    def _handle_tool_start(self, event: SessionEvent) -> None:
        """追加工具标题，并用临时状态展示运行中提示。"""

        self._stop_status_for_visible_output()
        self._start_status(f"{event.tool_name}({event.args})")

    def _handle_tool_update(self, event: SessionEvent) -> None:
        """追加工具过程更新，避免长任务期间完全无反馈。"""

        self._stop_status_for_visible_output()
        self._ensure_tool_title(event)
        self._console.print(tool_update(event.result, event.is_error), end="")

    def _handle_tool_end(self, event: SessionEvent) -> None:
        """追加工具最终结果摘要。"""

        self._stop_status_for_visible_output()
        self._ensure_tool_title(event)
        self._console.print(tool_result_summary(event.result, event.is_error), end="")

    def _finish_turn(self) -> None:
        """结束当前回合并输出最终耗时。"""

        self.stop_status()
        self._ensure_newline()
        self._console.print(elapsed_block(self._final_elapsed_seconds()), end="")
        self._turn_start = 0.0
        self._assistant_has_output = False
        self._assistant_has_text_output = False
        self._assistant_last_char = ""
        self._assistant_last_kind = ""
        self._assistant_text_chunks.clear()
        self._pending_block_gap = False
        self._rendered_tool_titles.clear()

    def _start_status(self, message: str = "Working...") -> None:
        """显示或更新一条临时状态，状态内容不进入最终 scrollback。"""

        if self._status is not None:
            self._status.update(message)
            return
        self._status = self._console.status(message, spinner="dots")
        self._status.__enter__()

    def _stop_status_for_visible_output(self) -> None:
        """可见输出即将写入 scrollback 前停止临时状态。"""

        self.stop_status()

    def _mark_assistant_output(self, text: str, kind: str) -> None:
        """记录 assistant 当前段落是否已有可见字符以及最后一个字符。"""

        if not text:
            return
        self._assistant_has_output = True
        if kind == "text":
            self._assistant_has_text_output = True
        self._assistant_last_char = text[-1]
        self._assistant_last_kind = kind

    def _mark_assistant_markdown_output(self) -> None:
        """记录 Markdown 正文已经渲染，Rich Markdown 自身会以换行结尾。"""

        self._assistant_has_output = True
        self._assistant_has_text_output = True
        self._assistant_last_char = "\n"
        self._assistant_last_kind = "text"

    def _render_assistant_text(self, final_text: str) -> None:
        """把缓存或最终正文作为完整 Markdown 渲染一次。"""

        text = "".join(self._assistant_text_chunks) or final_text
        self._assistant_text_chunks.clear()
        if not text or self._assistant_has_text_output:
            return
        self._stop_status_for_visible_output()
        self._ensure_text_gap_after_thinking()
        self._console.print(assistant_markdown(text), end="", soft_wrap=True)
        self._mark_assistant_markdown_output()

    def _ensure_text_gap_after_thinking(self) -> None:
        """thinking 后接正文时补成独立段落，避免两种内容粘在同一行。"""

        if self._assistant_last_kind != "thinking":
            return
        self._ensure_newline()
        self._console.print("")
        self._assistant_last_char = "\n"

    def _ensure_tool_title(self, event: SessionEvent) -> None:
        """在工具输出前补一次工具标题，避免并发 start 事件和结果错位。"""

        key = self._tool_event_key(event)
        if key in self._rendered_tool_titles:
            return
        self._ensure_block_gap()
        self._console.print(tool_start(event.tool_name, event.args), end="")
        self._rendered_tool_titles.add(key)
        self._assistant_has_output = False
        self._assistant_has_text_output = False
        self._assistant_last_char = ""
        self._assistant_last_kind = ""
        self._assistant_text_chunks.clear()

    def _tool_event_key(self, event: SessionEvent) -> str:
        """返回本次工具调用的稳定标识，测试事件缺省 id 时用标题兜底。"""

        if event.tool_call_id:
            return event.tool_call_id
        return f"{event.tool_name}({event.args})"

    def _ensure_newline(self) -> None:
        """如果 assistant 正文没有以换行结束，补一个换行分隔后续块。"""

        if self._assistant_has_output and self._assistant_last_char != "\n":
            self._console.print("")
            self._assistant_last_char = "\n"

    def _ensure_pending_block_gap(self) -> None:
        """在 Working 或下一个可见内容块前消费一次空行间距。"""

        if not self._pending_block_gap:
            return
        self._console.print("")
        self._pending_block_gap = False

    def _ensure_block_gap(self) -> None:
        """在工具或错误块前保证和上一段 assistant/user 输出隔开。"""

        if self._assistant_has_output:
            self._ensure_newline()
            self._console.print("")
            self._assistant_has_output = False
            self._assistant_last_char = ""
            return
        self._ensure_pending_block_gap()

    def _final_elapsed_seconds(self) -> int:
        """返回最终展示耗时，短请求也至少显示 1 秒。"""

        if not self._turn_start:
            return 0
        return max(1, math.ceil(self._clock() - self._turn_start))
