from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agentcode.config import MemoryConfig, ProviderConfig
from agentcode.context import ContextSettings
from agentcode.llm import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    DoneStopReason,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    ToolCall,
    Usage,
    message_text,
)
from agentcode.permission import PermissionPolicy, load_permission_config
from agentcode.terminal.app import PROMPT_STYLE, TerminalApp, TerminalRenderer
from agentcode.tool import BaseTool, ExecutionMode, Registry, ToolResult, ToolUpdate, text_result


@pytest.mark.asyncio
async def test_single_provider_enters_prompt_loop_and_exits() -> None:
    """单 provider 不要求选择，直接进入对话输入循环。"""

    output = io.StringIO()
    provider = FakeProvider([_assistant_events(text="ok")])
    prompt = FakePrompt(["hi", "/exit"])
    app = TerminalApp(
        [_provider("Only", "openai")],
        Registry(),
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        clock=Clock([10.0, 10.1]),
        permission_policy=_permission_policy(),
    )

    await app.run_async()

    assert prompt.prompts == ["❯ ", "❯ "]
    assert prompt.bottom_toolbars == [
        "\npermission: default · model: fake-model",
        "\npermission: default · model: fake-model",
    ]
    assert "erase_when_done" not in prompt.kwargs[0]
    assert "erase_when_done" not in prompt.kwargs[1]
    assert prompt.kwargs[0]["style"] is PROMPT_STYLE
    assert prompt.kwargs[1]["style"] is PROMPT_STYLE
    assert provider.requests[0].messages[0].content == "hi"
    rendered = output.getvalue()
    assert "permission: default · model: fake-model" not in rendered
    assert "● hi" in rendered
    assert "● hi\n\nok" in rendered
    assert "ok" in rendered
    assert rendered.count("ok") == 1
    assert "耗时：1s\n\n" in rendered


@pytest.mark.asyncio
async def test_multiple_provider_selection_retries_invalid_input() -> None:
    """多 provider 会展示编号，并在非法编号后继续等待选择。"""

    output = io.StringIO()
    first = _provider("One", "openai")
    second = _provider("Two", "anthropic")
    provider = FakeProvider([])
    prompt = FakePrompt(["x", "2", "/exit"])
    app = TerminalApp(
        [first, second],
        Registry(),
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda cfg: provider.with_name(cfg.name, cfg.model),
        permission_policy=_permission_policy(),
    )

    await app.run_async()

    assert prompt.prompts == ["provider> ", "provider> ", "❯ "]
    assert prompt.bottom_toolbars == [None, None, "\npermission: default · model: two-model"]
    assert all("erase_when_done" not in kwargs for kwargs in prompt.kwargs)
    assert "style" not in prompt.kwargs[0]
    assert "style" not in prompt.kwargs[1]
    assert prompt.kwargs[2]["style"] is PROMPT_STYLE
    assert app.provider is provider
    assert provider.name == "Two"
    rendered = output.getvalue()
    assert "请选择 provider：\n\n1. One · one-model\n\n2. Two · two-model\n\n" in rendered
    assert "1. One · one-model" in rendered
    assert "2. Two · two-model" in rendered
    assert "请输入有效的 provider 编号。" in rendered


@pytest.mark.asyncio
async def test_prompt_eof_exits_without_turn() -> None:
    """主输入读取到 EOF 时安静退出，不提交空回合。"""

    output = io.StringIO()
    provider = FakeProvider([])
    prompt = FakePrompt([EOFError()])
    app = TerminalApp(
        [_provider("Only", "openai")],
        Registry(),
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        permission_policy=_permission_policy(),
    )

    await app.run_async()

    assert provider.requests == []


@pytest.mark.asyncio
async def test_tool_events_are_appended_to_scrollback() -> None:
    """工具标题、过程更新和最终结果会追加，临时 Running 不进入 scrollback。"""

    output = io.StringIO()
    registry = Registry()
    registry.register(UpdatingTool())
    provider = FakeProvider(
        [
            _assistant_events(
                text="先读",
                tool_calls=[ToolCall(id="call_1", name="read", arguments={"path": "a"})],
            ),
            _assistant_events(text="完成"),
        ]
    )
    prompt = FakePrompt(["read", "/exit"])
    app = TerminalApp(
        [_provider("Only", "openai")],
        registry,
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        clock=Clock([1.0, 1.1]),
        permission_policy=_permission_policy(),
    )

    await app.run_async()

    rendered = output.getvalue()
    lines = [line.rstrip() for line in rendered.splitlines()]
    assert "先读" in rendered
    assert "先读\n\n" in rendered
    assert '● read({"path":"a"})' in rendered
    assert rendered.count('● read({"path":"a"})') == 1
    assert "Running..." not in rendered
    assert "↳ partial" in rendered
    assert "⎿ file content" in rendered
    complete_index = lines.index("完成")
    assert lines[complete_index - 1] == ""
    assert "完成" in rendered


def test_renderer_does_not_duplicate_final_streamed_text() -> None:
    """流式正文会即时输出，并且 message_end 不重复渲染。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([100.0, 100.01]))
    message = AssistantMessage(content=[TextContent(text="hello")])

    renderer.render(_event("agent_start"))
    renderer.render(_event("message_update", text="hel"))
    renderer.render(_event("message_update", text="lo"))

    assert output.getvalue() == "hello"

    renderer.render(_event("message_end", message=message))
    renderer.render(_event("agent_end"))

    rendered = output.getvalue()
    assert rendered.count("hello") == 1
    assert "耗时：1s\n\n" in rendered


def test_renderer_prints_final_text_when_provider_did_not_stream_delta() -> None:
    """provider 没有产生 text delta 时，message_end 会兜底输出最终文本。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([100.0, 100.01]))
    message = AssistantMessage(content=[TextContent(text="fallback")])

    renderer.render(_event("agent_start"))
    renderer.render(_event("message_start", message=AssistantMessage()))
    renderer.render(_event("message_end", message=message))
    renderer.render(_event("agent_end"))

    assert "fallback" in output.getvalue()


def test_renderer_renders_assistant_text_as_markdown() -> None:
    """没有正文 delta 时，assistant 正文在 message_end 兜底按 Markdown 渲染。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([100.0, 100.01]))
    text = "# 标题\n\n- A\n\n```python\nprint(1)\n```"
    message = AssistantMessage(content=[TextContent(text=text)])

    renderer.render(_event("agent_start"))

    assert output.getvalue() == ""

    renderer.render(_event("message_end", message=message))

    rendered = output.getvalue()
    lines = [line.rstrip() for line in rendered.splitlines()]
    assert "标题" in lines
    assert "# 标题" not in rendered
    assert "```" not in rendered
    assert " • A" in lines
    assert any(line.strip() == "print(1)" for line in lines)


def test_renderer_separates_thinking_from_text() -> None:
    """thinking 和正式正文之间补空行，避免两种内容粘在一行。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([100.0, 100.01]))
    message = AssistantMessage(content=[TextContent(text="正式回复")])

    renderer.render(_event("agent_start"))
    renderer.render(_event("message_update", thinking="先分析"))
    renderer.render(_event("message_update", text="正式回复"))
    renderer.render(_event("message_end", message=message))

    assert "先分析\n\n正式回复" in output.getvalue()


def test_renderer_prints_final_text_after_thinking_without_streamed_text() -> None:
    """只有 thinking delta 时，message_end 仍会兜底输出最终正文。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([100.0, 100.01]))
    message = AssistantMessage(content=[TextContent(text="最终正文")])

    renderer.render(_event("agent_start"))
    renderer.render(_event("message_update", thinking="先分析"))
    renderer.render(_event("message_end", message=message))

    assert "先分析\n\n最终正文" in output.getvalue()


def test_renderer_separates_parallel_tool_results() -> None:
    """连续工具 start 不写入 scrollback，结果到来时按工具各自成块输出。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([100.0, 100.01]))

    renderer.render(_event("agent_start"))
    renderer.render(
        _event(
            "tool_execution_start",
            tool_call_id="call_1",
            tool_name="read",
            args='{"path":"README.md"}',
        )
    )
    renderer.render(
        _event(
            "tool_execution_start",
            tool_call_id="call_2",
            tool_name="ls",
            args='{"path":".agentcode"}',
        )
    )
    renderer.render(
        _event(
            "tool_execution_end",
            tool_call_id="call_1",
            tool_name="read",
            args='{"path":"README.md"}',
            result="# AgentCode",
        )
    )
    renderer.render(
        _event(
            "tool_execution_end",
            tool_call_id="call_2",
            tool_name="ls",
            args='{"path":".agentcode"}',
            result="config.yaml",
        )
    )

    rendered = output.getvalue()
    lines = [line.rstrip() for line in rendered.splitlines()]
    assert rendered.count("● read") == 1
    assert rendered.count("● ls") == 1
    assert lines == [
        '● read({"path":"README.md"})',
        "  ⎿ # AgentCode",
        "",
        '● ls({"path":".agentcode"})',
        "  ⎿ config.yaml",
        "",
    ]


def test_renderer_status_stops_before_visible_output() -> None:
    """正文增量渲染为可见输出前，临时 Working 状态会被关闭。"""

    output = io.StringIO()
    renderer = TerminalRenderer(_console(output), Clock([1.0, 1.1]))
    message = AssistantMessage(content=[TextContent(text="ok")])

    renderer.render(_event("agent_start"))
    renderer.render(_event("message_end", message=_user_message("hi")))
    assert renderer._status is not None  # noqa: SLF001
    assert output.getvalue() == "● hi\n\n"

    renderer.render(_event("message_update", text="ok"))
    assert renderer._status is None  # noqa: SLF001
    assert output.getvalue() == "● hi\n\nok"

    renderer.render(_event("message_end", message=message))

    assert renderer._status is None  # noqa: SLF001
    assert "Working..." not in output.getvalue()
    assert "ok" in output.getvalue()


def test_permission_mode_cycles_and_updates_toolbar() -> None:
    """Shift+Tab 绑定背后的切换逻辑会循环更新状态栏。"""

    app = TerminalApp(
        [_provider("Only", "openai")],
        Registry(),
        console=_console(io.StringIO()),
        prompt_reader=FakePrompt([]),
        provider_factory=lambda _: FakeProvider([]),
        permission_policy=_permission_policy(),
    )
    app.provider = FakeProvider([])

    app._cycle_permission_mode()  # noqa: SLF001
    assert app._bottom_toolbar() == "\npermission: acceptEdits · model: fake-model"
    app._cycle_permission_mode()  # noqa: SLF001
    assert app._bottom_toolbar() == "\npermission: plan · model: fake-model"
    app._cycle_permission_mode()  # noqa: SLF001
    assert app._bottom_toolbar() == "\npermission: bypassPermissions · model: fake-model"
    app._cycle_permission_mode()  # noqa: SLF001
    assert app._bottom_toolbar() == "\npermission: default · model: fake-model"


@pytest.mark.asyncio
async def test_permission_ask_menu_allows_once(tmp_path: Path) -> None:
    """命令工具在 default 模式下会弹出审批菜单，允许本次后才执行。"""

    output = io.StringIO()
    registry = Registry()
    registry.register(UpdatingBashTool())
    provider = FakeProvider(
        [
            _assistant_events(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="bash",
                        arguments={"command": "echo hi"},
                    )
                ],
            ),
            _assistant_events(text="完成"),
        ]
    )
    prompt = FakePrompt(["run", "1", "/exit"])
    app = TerminalApp(
        [_provider("Only", "openai")],
        registry,
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        permission_policy=_permission_policy(tmp_path),
    )

    await app.run_async()

    rendered = output.getvalue()
    assert "+-- 权限确认" in rendered
    assert "工具: Bash" in rendered
    assert "Esc/Ctrl+C 取消\n\n" in rendered
    assert "Esc/Ctrl+C 取消\n\n\n" not in rendered
    assert "⎿ bash ok" in rendered
    assert prompt.prompts == ["❯ ", "permission> ", "❯ "]


@pytest.mark.asyncio
async def test_plan_and_do_commands_switch_modes(tmp_path: Path) -> None:
    """`/plan` 只暴露只读工具，`/do` 固定切回 default 并提交执行提示。"""

    output = io.StringIO()
    registry = Registry()
    registry.register(UpdatingTool())
    registry.register(UpdatingBashTool())
    provider = FakeProvider(
        [
            _assistant_events(text="计划"),
            _assistant_events(text="执行"),
        ]
    )
    prompt = FakePrompt(["/plan", "think", "/do", "/exit"])
    app = TerminalApp(
        [_provider("Only", "openai")],
        registry,
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        permission_policy=_permission_policy(tmp_path),
    )

    await app.run_async()

    assert provider.requests[0].tools is not None
    assert [tool.name for tool in provider.requests[0].tools] == ["read"]
    assert provider.requests[1].messages[-1].content == "按计划执行。"
    assert prompt.bottom_toolbars == [
        "\npermission: default · model: fake-model",
        "\npermission: plan · model: fake-model",
        "\npermission: plan · model: fake-model",
        "\npermission: default · model: fake-model",
    ]


@pytest.mark.asyncio
async def test_compact_command_runs_manual_compaction(tmp_path: Path) -> None:
    """`/compact` 使用当前会话历史生成摘要并输出压缩结果。"""

    output = io.StringIO()
    provider = FakeProvider(
        [
            _assistant_events(text="ok"),
            _assistant_events(text="摘要"),
        ]
    )
    prompt = FakePrompt(["hi", "/compact focus", "/exit"])
    app = TerminalApp(
        [_provider("Only", "openai")],
        Registry(),
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        permission_policy=_permission_policy(tmp_path),
        context_settings=ContextSettings(keep_recent_tokens=1, artifact_root=tmp_path),
    )

    await app.run_async()

    rendered = output.getvalue()
    assert "compacting: manual" in rendered
    assert "compacted:" in rendered
    assert "摘要" not in rendered
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_resume_command_restores_session_history(tmp_path: Path) -> None:
    """`/resume` 选中历史后，下一轮请求会带上恢复出的消息。"""

    output = io.StringIO()
    provider = FakeProvider(
        [
            _assistant_events(text="ok"),
            _assistant_events(text="again"),
        ]
    )
    prompt = FakePrompt(["hi", "/resume", "1", "next", "/exit"])
    app = TerminalApp(
        [_provider("Only", "openai")],
        Registry(),
        console=_console(output),
        prompt_reader=prompt,
        provider_factory=lambda _: provider,
        permission_policy=_permission_policy(tmp_path),
        project_root=tmp_path,
        memory_config=MemoryConfig(
            session_dir=str(tmp_path / "sessions"),
            notes_dir=str(tmp_path / "notes"),
            auto_notes=False,
        ),
    )

    await app.run_async()

    assert "已恢复会话" in output.getvalue()
    assert prompt.prompts == ["❯ ", "❯ ", "resume> ", "❯ ", "❯ "]
    assert [message_text(message) for message in provider.requests[1].messages] == [
        "hi",
        "ok",
        "next",
    ]


class FakePrompt:
    """测试用 prompt_reader，按脚本返回输入或抛出异常。"""

    def __init__(self, inputs: list[str | BaseException]) -> None:
        """保存输入脚本并记录每次 prompt 文案。"""

        self._inputs = inputs
        self.prompts: list[str] = []
        self.bottom_toolbars: list[object] = []
        self.kwargs: list[dict[str, object]] = []

    async def prompt_async(self, message: str, **kwargs: object) -> str:
        """返回下一条脚本输入，异常对象会按原样抛出。"""

        self.prompts.append(message)
        bottom_toolbar = kwargs.get("bottom_toolbar")
        if callable(bottom_toolbar):
            bottom_toolbar = bottom_toolbar()
        self.bottom_toolbars.append(bottom_toolbar)
        self.kwargs.append(kwargs)
        if not self._inputs:
            raise EOFError
        value = self._inputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeProvider:
    """测试用 Provider，按脚本产出统一 assistant 事件。"""

    def __init__(self, scripts: list[list[AssistantMessageEvent]]) -> None:
        """保存事件脚本并初始化 provider 展示字段。"""

        self.api = "fake"
        self.name = "Fake"
        self.model = "fake-model"
        self.context_window = 100_000
        self._scripts = scripts
        self.requests: list[Context] = []
        self.stream_options: list[StreamOptions | None] = []

    def with_name(self, name: str, model: str) -> "FakeProvider":
        """调整展示字段并返回自身，方便 provider_factory 测试选择结果。"""

        self.name = name
        self.model = model
        return self

    async def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """记录请求并产出下一组脚本事件。"""

        self.requests.append(context)
        self.stream_options.append(options)
        script = self._scripts.pop(0)
        for event in script:
            yield event


class UpdatingTool(BaseTool):
    """测试用工具，会先发过程更新再返回最终文本。"""

    def name(self) -> str:
        """返回工具名。"""

        return "read"

    def description(self) -> str:
        """返回 provider tools 中使用的工具说明。"""

        return "read"

    def parameters(self) -> dict[str, object]:
        """返回空参数 schema，测试中不依赖校验细节。"""

        return {"type": "object", "properties": {}}

    def execution_mode(self) -> ExecutionMode:
        """声明为串行工具，方便测试输出顺序稳定。"""

        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """发送一次过程更新并返回最终工具结果。"""

        if on_update is not None:
            on_update(text_result("partial"))
        return text_result("file content")


class UpdatingBashTool(BaseTool):
    """测试用 bash 工具，记录审批通过后的执行结果。"""

    def name(self) -> str:
        """返回工具名。"""

        return "bash"

    def description(self) -> str:
        """返回 provider tools 中使用的工具说明。"""

        return "bash"

    def parameters(self) -> dict[str, object]:
        """声明 bash 测试工具接受 command 参数。"""

        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    def execution_mode(self) -> ExecutionMode:
        """声明命令工具必须串行执行。"""

        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """返回固定结果，证明审批后执行了真实工具。"""

        return text_result("bash ok")


class Clock:
    """测试用时钟，按顺序返回固定时间。"""

    def __init__(self, values: list[float]) -> None:
        """保存待返回时间序列。"""

        self._values = values

    def __call__(self) -> float:
        """返回下一项时间，序列耗尽后复用最后一项。"""

        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


def _console(output: io.StringIO) -> Console:
    """创建测试用 Rich Console，关闭颜色以便断言纯文本。"""

    return Console(file=output, force_terminal=False, color_system=None, width=120)


def _permission_policy(root: Path | None = None) -> PermissionPolicy:
    """创建不读取用户真实配置的测试权限策略。"""

    project_root = root or Path.cwd()
    missing = project_root / "missing-permissions.yaml"
    return PermissionPolicy(
        project_root,
        load_permission_config(
            project_root,
            user_path=missing,
            project_path=missing,
            local_path=project_root / "permissions.local.yaml",
        ),
    )


def _provider(name: str, protocol: str) -> ProviderConfig:
    """创建测试用 provider 配置。"""

    return ProviderConfig(
        name=name,
        protocol=protocol,  # type: ignore[arg-type]
        api_key="test-key",
        model=f"{name.lower()}-model",
    )


def _assistant_events(
    text: str = "",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> list[AssistantMessageEvent]:
    """创建测试用 assistant 事件流。"""

    content: list[AssistantContent] = []
    events: list[AssistantMessageEvent] = [StartEvent(partial=AssistantMessage())]
    if text:
        content.append(TextContent(text=text))
        events.append(
            TextDeltaEvent(
                content_index=len(content) - 1,
                delta=text,
                partial=AssistantMessage(content=list(content)),
            )
        )
    content.extend(tool_calls or [])
    reason: DoneStopReason = "toolUse" if tool_calls else "stop"
    message = AssistantMessage(
        content=content,
        api="fake",
        provider="Fake",
        model="fake-model",
        usage=usage or Usage(),
        stop_reason=reason,
    )
    events.append(DoneEvent(reason=reason, message=message))
    return events


def _event(event_type: str, **kwargs: object) -> object:
    """创建局部测试用 SessionEvent，避免每个断言重复导入。"""

    from agentcode.session import SessionEvent

    return SessionEvent(type=event_type, **kwargs)  # type: ignore[arg-type]


def _user_message(text: str) -> object:
    """创建测试用 UserMessage。"""

    from agentcode.llm import UserMessage

    return UserMessage(content=text)
