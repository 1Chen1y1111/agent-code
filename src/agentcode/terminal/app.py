"""AgentCode 的普通终端 CLI 应用。

负责 provider 选择、prompt_toolkit 输入循环，以及把 Session 事件追加渲染到终端 scrollback。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from pathlib import Path
import time
from typing import Protocol, cast

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.status import Status
from rich.text import Text

from agentcode import __version__
from agentcode.config import MemoryConfig, ProviderConfig
from agentcode.context import ContextSettings
from agentcode.llm import Message, Provider, create_provider, message_text
from agentcode.memory import MemoryExtractor, MemoryStore
from agentcode.mcp import McpManager, McpServerConfig
from agentcode.permission import (
    PermissionApproval,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    VALID_MODES,
    next_permission_mode,
)
from agentcode.prompt import PromptBuildOptions, render_banner
from agentcode.session import AgentSession, SessionEvent
from agentcode.session_store import SessionStore, StoredSessionInfo
from agentcode.tool import Registry, create_default_registry
from agentcode.terminal.commands import create_builtin_command_registry
from agentcode.terminal.view import (
    assistant_markdown,
    assistant_text_delta,
    compaction_end_block,
    compaction_start_block,
    elapsed_block,
    error_block,
    provider_option,
    thinking_delta,
    tool_result_summary,
    tool_start,
    tool_update,
    user_block,
)

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
        mcp_configs: Sequence[McpServerConfig] = (),
        mcp_manager_factory: Callable[
            [Sequence[McpServerConfig], Registry], McpManager
        ] = McpManager,
        clock: Callable[[], float] = time.monotonic,
        permission_policy: PermissionPolicy | None = None,
        context_settings: ContextSettings | None = None,
        project_root: str | Path | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        """保存启动依赖；真实 provider 和 session 会在用户选定后创建。"""

        self._project_root = Path(project_root or Path.cwd()).resolve()
        self.providers = providers
        self._registry = registry or create_default_registry()
        self._prompt_options = prompt_options or PromptBuildOptions()
        self._console = console or Console()
        self._prompt_reader = prompt_reader or cast(
            PromptReader,
            PromptSession(
                history=InMemoryHistory(),
                erase_when_done=True,
            ),
        )
        self._provider_factory = provider_factory
        self._mcp_configs = tuple(mcp_configs)
        self._mcp_manager_factory = mcp_manager_factory
        self._clock = clock
        self._permission_policy = permission_policy or PermissionPolicy.load(
            self._project_root
        )
        self._context_settings = context_settings
        self._memory_config = memory_config
        self._session_store: SessionStore | None = None
        self._memory_store: MemoryStore | None = None
        self._memory_extractor: MemoryExtractor | None = None
        self._permission_mode = self._permission_policy.default_mode()
        self._slash_commands = create_builtin_command_registry()
        self._slash_command_context = _TerminalCommandContext(self)
        self._key_bindings = self._build_key_bindings()
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

        mcp_manager: McpManager | None = None
        mcp_tool_count = 0
        try:
            if self._mcp_configs:
                mcp_manager = self._mcp_manager_factory(
                    self._mcp_configs, self._registry
                )
                mcp_tool_count = await mcp_manager.start()

            self._console.print(
                render_banner(__version__, str(self._project_root), mcp_tool_count),
                end="",
            )
            provider_config = await self._select_provider_config()
            if provider_config is None:
                return

            self.provider = self._provider_factory(provider_config)
            self._initialize_memory_runtime()
            session_id = self._create_stored_session()
            self.agent_session = self._create_agent_session(session_id=session_id)

            while True:
                try:
                    raw_text = await self._prompt_reader.prompt_async(
                        PROMPT_TEXT,
                        bottom_toolbar=self._bottom_toolbar,
                        refresh_interval=0.2,
                        style=PROMPT_STYLE,
                        key_bindings=self._key_bindings,
                    )
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break

                message = raw_text.strip()
                if not message:
                    continue
                command_outcome = await self._slash_commands.dispatch(
                    message,
                    self._slash_command_context,
                )
                if command_outcome == "exit":
                    break
                if command_outcome == "handled":
                    continue
                await self._run_turn(message)
        finally:
            if mcp_manager is not None:
                await mcp_manager.close()

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

            selected_provider = self._provider_from_selection(selection)
            if selected_provider is not None:
                return selected_provider
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

    def _initialize_memory_runtime(self) -> None:
        """按配置初始化会话存档和自动笔记运行时。"""

        config = self._memory_config
        if config is None or not config.enabled:
            return
        session_dir = config.session_dir or ".agentcode/sessions"
        notes_dir = config.notes_dir or ".agentcode/memory"
        self._session_store = SessionStore(session_dir, project_root=self._project_root)
        self._session_store.cleanup_expired(config.retention_days)
        self._memory_store = MemoryStore(notes_dir, project_root=self._project_root)
        self._memory_extractor = MemoryExtractor(max_tokens=config.note_max_tokens)

    def _create_stored_session(self) -> str | None:
        """创建本次运行的 JSONL session，未启用存档时返回 None。"""

        if self._session_store is None or self.provider is None:
            return None
        return self._session_store.create(
            provider=self.provider.name,
            model=self.provider.model,
        )

    def _create_agent_session(
        self,
        *,
        session_id: str | None,
        initial_messages: list[Message] | None = None,
    ) -> AgentSession:
        """创建绑定当前 provider 和运行时依赖的 AgentSession。"""

        if self.provider is None:
            raise RuntimeError("provider 未初始化")
        return AgentSession(
            self.provider,
            self._registry,
            self._prompt_options,
            permission_policy=self._permission_policy,
            permission_mode=lambda: self._permission_mode,
            permission_approver=self._approve_permission,
            context_settings=self._context_settings,
            project_root=self._project_root,
            session_id=session_id,
            session_store=self._session_store,
            memory_store=self._memory_store,
            memory_extractor=self._memory_extractor,
            auto_notes=(
                bool(self._memory_config and self._memory_config.auto_notes)
                and self._memory_store is not None
            ),
            initial_messages=initial_messages,
        )

    def _bottom_toolbar(self) -> str:
        """生成 prompt_toolkit 底部状态栏文本，显示当前权限模式和模型。"""

        return self._bottom_toolbar_for_text(_current_prompt_text())

    def _bottom_toolbar_for_text(self, text: str) -> str:
        """根据当前输入生成底部栏文本。"""

        command_toolbar = self._slash_commands.toolbar_text(text)
        if command_toolbar is not None:
            return command_toolbar
        if self.provider is None:
            return ""
        return f"\npermission: {self._permission_mode} · model: {self.provider.model}"

    def _build_key_bindings(self) -> KeyBindings:
        """创建主输入框按键绑定，用 Shift+Tab 循环切换权限模式。"""

        bindings = KeyBindings()

        @bindings.add("s-tab")
        @bindings.add(Keys.BackTab)
        @bindings.add("escape", "[", "Z")
        def _(event: object) -> None:
            """处理 Shift+Tab 权限模式切换。"""

            self._cycle_permission_mode()
            app = getattr(event, "app", None)
            if app is not None:
                app.invalidate()

        @bindings.add("tab")
        def _(event: object) -> None:
            """处理斜杠命令 Tab 补全。"""

            buffer = getattr(event, "current_buffer", None)
            if buffer is None:
                return
            text = str(getattr(buffer, "text", "") or "")
            completed = self._slash_commands.complete_text(text)
            if completed is None:
                return
            setattr(buffer, "text", completed)
            setattr(buffer, "cursor_position", len(completed))
            app = getattr(event, "app", None)
            if app is not None:
                app.invalidate()

        return bindings

    def _cycle_permission_mode(self) -> None:
        """把当前权限模式切换到循环序列中的下一档。"""

        self._permission_mode = next_permission_mode(self._permission_mode)

    def _set_permission_mode(self, mode: PermissionMode) -> None:
        """把当前权限模式设置为命令指定值。"""

        self._permission_mode = mode

    def _write_command_output(self, text: str, *, style: str | None = None) -> None:
        """向终端追加本地命令输出。"""

        self._renderer.stop_status()
        self._console.print(
            Text(text, style=style or ""),
            end="" if text.endswith("\n") else "\n",
        )

    def _session_status_text(self) -> str:
        """生成 `/session` 的当前会话状态文本。"""

        provider = self.provider
        session = self.agent_session
        session_id = session.session_id if session is not None else "(none)"
        session_file = (
            str(self._session_store.path_for(session_id))
            if self._session_store is not None and session is not None
            else "(disabled)"
        )
        provider_name = provider.name if provider is not None else "(none)"
        model = provider.model if provider is not None else "(none)"
        message_count = len(session.messages()) if session is not None else 0
        return (
            "+-- session ---------------------------------\n"
            f"| id: {session_id}\n"
            f"| provider: {provider_name}\n"
            f"| model: {model}\n"
            f"| messages: {message_count}\n"
            f"| archive: {session_file}\n"
            "+--------------------------------------------\n"
        )

    def _memory_status_text(self) -> str:
        """生成 `/memory` 的当前记忆状态文本。"""

        config = self._memory_config
        enabled = bool(config and config.enabled and self._memory_store is not None)
        auto_notes = bool(config and config.auto_notes and enabled)
        notes_count = 0
        notes_dir = "(disabled)"
        user_notes_dir = "(disabled)"
        if self._memory_store is not None:
            notes_count = len(self._memory_store.load_all())
            notes_dir = str(self._memory_store.notes_dir)
            user_notes_dir = str(self._memory_store.user_notes_dir)
        session_dir = (
            str(self._session_store.root)
            if self._session_store is not None
            else "(disabled)"
        )
        return (
            "+-- memory ----------------------------------\n"
            f"| enabled: {enabled}\n"
            f"| auto notes: {auto_notes}\n"
            f"| notes: {notes_count}\n"
            f"| notes dir: {notes_dir}\n"
            f"| user notes dir: {user_notes_dir}\n"
            f"| sessions dir: {session_dir}\n"
            "+--------------------------------------------\n"
        )

    def _permission_status_text(self) -> str:
        """生成 `/permissions` 的当前权限状态文本。"""

        return (
            "+-- permissions -----------------------------\n"
            f"| current: {self._permission_mode}\n"
            f"| modes: {', '.join(VALID_MODES)}\n"
            "| shortcut: Shift+Tab cycles modes\n"
            "+--------------------------------------------\n"
        )

    def _tools_status_text(self) -> str:
        """生成 `/tools` 的当前工具状态文本。"""

        names = self._registry.names()
        tool_lines = "\n".join(f"| - {name}" for name in names) if names else "| (none)"
        return (
            "+-- tools -----------------------------------\n"
            f"{tool_lines}\n"
            "+--------------------------------------------\n"
        )

    async def _approve_permission(
        self,
        request: PermissionRequest,
    ) -> PermissionApproval:
        """在工具执行前向用户确认 Ask 权限请求。"""

        self._renderer.stop_status()
        self._console.print(_permission_request_block(request), end="")
        selection = {"index": 0}
        labels = ("允许本次", "永久允许", "拒绝本次")
        bindings = _permission_key_bindings(selection)

        def toolbar() -> str:
            """展示当前审批菜单选项，供方向键切换时刷新。"""

            return f"\n当前选择: {selection['index'] + 1}. {labels[selection['index']]}"

        try:
            raw = await self._prompt_reader.prompt_async(
                "permission> ",
                bottom_toolbar=toolbar,
                style=PROMPT_STYLE,
                key_bindings=bindings,
            )
        except (EOFError, KeyboardInterrupt):
            return "cancel"
        return _approval_from_input(raw, selection["index"])

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

    async def _run_compact(self, custom_instructions: str | None = None) -> None:
        """执行手动上下文压缩命令并把结果追加到终端。"""

        if self.agent_session is None:
            return

        self._renderer.stop_status()
        self._console.print(compaction_start_block("manual"), end="")
        try:
            report = await self.agent_session.compact(custom_instructions)
        except Exception as exc:  # noqa: BLE001 - 手动命令错误需要恢复终端并展示。
            self._console.print(error_block(exc), end="")
            return
        self._console.print(
            compaction_end_block("manual", report),
            end="",
        )

    async def _run_resume(self) -> None:
        """打开会话选择器并把当前 AgentSession 替换为选中的历史会话。"""

        if self._session_store is None:
            self._console.print("未启用会话存档，无法恢复历史。\n", style="dim")
            return
        sessions = self._session_store.list_sessions(limit=50)
        if not sessions:
            self._console.print("没有可恢复的历史会话。\n", style="dim")
            return
        selected = await self._select_resume_session(sessions)
        if selected is None:
            self._console.print("已取消恢复。\n", style="dim")
            return
        loaded = self._session_store.load(selected.session_id)
        self.agent_session = self._create_agent_session(
            session_id=loaded.session_id,
            initial_messages=loaded.messages,
        )
        self._console.print(f"已恢复会话 {loaded.session_id}。\n", style="dim")

    async def _select_resume_session(
        self,
        sessions: list[StoredSessionInfo],
    ) -> StoredSessionInfo | None:
        """用 prompt_toolkit 提供可搜索的会话选择入口。"""

        self._console.print(_resume_list_block(sessions), end="")
        selection = {"index": 0}
        bindings = _resume_key_bindings(selection, sessions)

        def toolbar() -> str:
            """展示当前 resume 选择器提示。"""

            index = min(selection["index"], max(0, len(sessions) - 1))
            current = sessions[index]
            return (
                "\n输入关键词过滤 · ↑/↓ 选择 · Enter 恢复 · Esc 取消 | "
                f"当前: {current.session_id}"
            )

        try:
            raw = await self._prompt_reader.prompt_async(
                "resume> ",
                bottom_toolbar=toolbar,
                style=PROMPT_STYLE,
                key_bindings=bindings,
            )
        except (EOFError, KeyboardInterrupt):
            return None
        value = raw.strip()
        if value == "cancel":
            return None
        return _resolve_resume_selection(value, sessions, selection["index"])


class _TerminalCommandContext:
    """把 TerminalApp 的能力收窄成斜杠命令上下文。"""

    def __init__(self, app: TerminalApp) -> None:
        """保存当前终端应用实例。"""

        self._app = app

    def write(self, text: str, *, style: str | None = None) -> None:
        """向终端输出本地命令结果。"""

        self._app._write_command_output(text, style=style)

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """切换当前权限模式。"""

        self._app._set_permission_mode(mode)

    async def run_turn(self, message: str) -> None:
        """把消息提交给 Agent 执行。"""

        await self._app._run_turn(message)

    async def run_compact(self, custom_instructions: str | None = None) -> None:
        """触发手动上下文压缩。"""

        await self._app._run_compact(custom_instructions)

    async def run_resume(self) -> None:
        """触发会话恢复流程。"""

        await self._app._run_resume()

    def session_status(self) -> str:
        """返回当前会话状态文本。"""

        return self._app._session_status_text()

    def memory_status(self) -> str:
        """返回当前记忆状态文本。"""

        return self._app._memory_status_text()

    def permission_status(self) -> str:
        """返回当前权限状态文本。"""

        return self._app._permission_status_text()

    def tools_status(self) -> str:
        """返回当前工具状态文本。"""

        return self._app._tools_status_text()


def _permission_request_block(request: PermissionRequest) -> str:
    """把权限确认请求格式化为终端中的 ASCII 确认块。"""

    preview = request.preview.replace("\n", "\\n")
    reason = request.reason.replace("\n", " ")
    return (
        "+-- 权限确认 --------------------------------\n"
        f"| 工具: {request.friendly_name}\n"
        f"| 参数: {preview}\n"
        f"| 原因: {reason}\n"
        "|\n"
        "| > 1. 允许本次\n"
        "|   2. 永久允许\n"
        "|   3. 拒绝本次\n"
        "+-------------------------------------------\n\n"
        "↑/↓ 选择 · Enter 确认 · 1/2/3 直选 · Esc/Ctrl+C 取消\n\n"
    )


def _permission_key_bindings(selection: dict[str, int]) -> KeyBindings:
    """创建权限确认菜单的方向键和数字键绑定。"""

    bindings = KeyBindings()

    @bindings.add("up")
    def _(event: object) -> None:
        """向上移动权限菜单光标。"""

        selection["index"] = (selection["index"] - 1) % 3
        app = getattr(event, "app", None)
        if app is not None:
            app.invalidate()

    @bindings.add("down")
    def _(event: object) -> None:
        """向下移动权限菜单光标。"""

        selection["index"] = (selection["index"] + 1) % 3
        app = getattr(event, "app", None)
        if app is not None:
            app.invalidate()

    @bindings.add("enter")
    def _(event: object) -> None:
        """确认当前高亮的权限菜单项。"""

        app = getattr(event, "app", None)
        if app is not None:
            app.exit(result=str(selection["index"] + 1))

    @bindings.add("1")
    @bindings.add("2")
    @bindings.add("3")
    def _(event: object) -> None:
        """用数字键直接选择权限菜单项。"""

        app = getattr(event, "app", None)
        key_sequence = getattr(event, "key_sequence", ())
        if app is not None and key_sequence:
            app.exit(result=str(key_sequence[0].key))

    @bindings.add("escape")
    def _(event: object) -> None:
        """取消权限确认，并让 Agent 停止当前轮。"""

        app = getattr(event, "app", None)
        if app is not None:
            app.exit(result="cancel")

    return bindings


def _approval_from_input(raw: str, selected_index: int) -> PermissionApproval:
    """把 prompt 输入或菜单默认选项转换为权限审批结果。"""

    value = raw.strip()
    if not value:
        value = str(selected_index + 1)
    if value == "cancel":
        return "cancel"
    if value == "2":
        return "allow_always"
    if value == "3":
        return "deny_once"
    return "allow_once"


def _resume_list_block(sessions: list[StoredSessionInfo]) -> str:
    """把最近会话列表格式化为 ASCII 块。"""

    lines = ["+-- resume ----------------------------------"]
    for index, session in enumerate(sessions[:10], start=1):
        title = session.first_user_message or "(empty)"
        lines.append(
            f"| {index:>2}. {session.modified_at:%Y-%m-%d %H:%M} "
            f"{session.session_id}  {title[:48]}"
        )
    if len(sessions) > 10:
        lines.append(f"| ... 还有 {len(sessions) - 10} 个会话，可输入关键词过滤")
    lines.append("+-------------------------------------------")
    lines.append("")
    return "\n".join(lines) + "\n"


def _resume_key_bindings(
    selection: dict[str, int],
    sessions: list[StoredSessionInfo],
) -> KeyBindings:
    """创建 resume 选择器的方向键、Enter 和取消绑定。"""

    bindings = KeyBindings()

    @bindings.add("up")
    def _(event: object) -> None:
        """向上移动 resume 选择器光标。"""

        selection["index"] = max(0, selection["index"] - 1)
        app = getattr(event, "app", None)
        if app is not None:
            app.invalidate()

    @bindings.add("down")
    def _(event: object) -> None:
        """向下移动 resume 选择器光标。"""

        selection["index"] = min(len(sessions) - 1, selection["index"] + 1)
        app = getattr(event, "app", None)
        if app is not None:
            app.invalidate()

    @bindings.add("enter")
    def _(event: object) -> None:
        """确认当前过滤结果中的选中会话。"""

        app = getattr(event, "app", None)
        if app is None:
            return
        buffer = getattr(app, "current_buffer", None)
        query = str(getattr(buffer, "text", "") or "")
        filtered = _filter_resume_sessions(query, sessions)
        if not filtered:
            app.exit(result="cancel")
            return
        index = min(selection["index"], len(filtered) - 1)
        app.exit(result=filtered[index].session_id)

    @bindings.add("escape")
    def _(event: object) -> None:
        """取消恢复会话。"""

        app = getattr(event, "app", None)
        if app is not None:
            app.exit(result="cancel")

    return bindings


def _resolve_resume_selection(
    value: str,
    sessions: list[StoredSessionInfo],
    selected_index: int,
) -> StoredSessionInfo | None:
    """把 resume prompt 返回值解析成会话。"""

    if not value:
        if not sessions:
            return None
        return sessions[min(selected_index, len(sessions) - 1)]
    for session in sessions:
        if session.session_id == value:
            return session
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
    filtered = _filter_resume_sessions(value, sessions)
    return filtered[0] if filtered else None


def _filter_resume_sessions(
    query: str,
    sessions: list[StoredSessionInfo],
) -> list[StoredSessionInfo]:
    """按 session id、模型和首条用户消息做简单过滤。"""

    text = query.strip().casefold()
    if not text:
        return sessions
    return [
        session
        for session in sessions
        if text in session.session_id.casefold()
        or text in session.model.casefold()
        or text in session.first_user_message.casefold()
    ]


def _current_prompt_text() -> str:
    """读取当前 prompt_toolkit 输入内容；测试或非交互上下文返回空串。"""

    app = get_app_or_none()
    if app is None:
        return ""
    buffer = getattr(app, "current_buffer", None)
    if buffer is None:
        return ""
    return str(getattr(buffer, "text", "") or "")


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
        if event.type == "compaction_start":
            self.stop_status()
            self._ensure_block_gap()
            self._console.print(
                compaction_start_block(event.compaction_reason or "threshold"),
                end="",
            )
            return
        if event.type == "compaction_end":
            self.stop_status()
            self._ensure_block_gap()
            self._console.print(
                compaction_end_block(
                    event.compaction_reason or "threshold",
                    event.compaction,
                    will_retry=event.will_retry,
                    error_message=event.error_message,
                ),
                end="",
            )
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
        """实时追加 thinking 和正文增量，避免非思考模式看起来无输出。"""

        if event.thinking:
            self._stop_status_for_visible_output()
            self._ensure_pending_block_gap()
            self._console.print(thinking_delta(event.thinking), end="")
            self._mark_assistant_output(event.thinking, "thinking")
        if event.text:
            self._stop_status_for_visible_output()
            self._ensure_pending_block_gap()
            self._ensure_text_gap_after_thinking()
            self._console.print(assistant_text_delta(event.text), end="")
            self._mark_assistant_output(event.text, "text")

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
