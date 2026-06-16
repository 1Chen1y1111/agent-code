from __future__ import annotations

import pytest

from agentcode.permission import PermissionMode
from agentcode.terminal.commands import (
    CommandOutcome,
    SlashCommand,
    SlashCommandRegistry,
    create_builtin_command_registry,
)


class FakeCommandContext:
    """测试用命令上下文，记录命令 handler 的副作用。"""

    def __init__(self) -> None:
        """初始化记录字段。"""

        self.outputs: list[str] = []
        self.styles: list[str | None] = []
        self.mode: PermissionMode = "default"
        self.turns: list[str] = []
        self.compactions: list[str | None] = []
        self.resume_count = 0

    def write(self, text: str, *, style: str | None = None) -> None:
        """记录本地命令输出。"""

        self.outputs.append(text)
        self.styles.append(style)

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """记录权限模式切换。"""

        self.mode = mode

    async def run_turn(self, message: str) -> None:
        """记录提交给 Agent 的消息。"""

        self.turns.append(message)

    async def run_compact(self, custom_instructions: str | None = None) -> None:
        """记录压缩参数。"""

        self.compactions.append(custom_instructions)

    async def run_resume(self) -> None:
        """记录恢复命令调用次数。"""

        self.resume_count += 1

    def session_status(self) -> str:
        """返回固定会话状态。"""

        return "session status\n"

    def memory_status(self) -> str:
        """返回固定记忆状态。"""

        return "memory status\n"

    def permission_status(self) -> str:
        """返回固定权限状态。"""

        return "permission status\n"

    def tools_status(self) -> str:
        """返回固定工具状态。"""

        return "tools status\n"


async def _noop_handler(
    context: FakeCommandContext,
    args: str,
) -> CommandOutcome:
    """测试用空命令 handler。"""

    return "handled"


def test_registry_rejects_command_name_conflicts() -> None:
    """注册表会在启动期拒绝命令名和别名冲突。"""

    registry = SlashCommandRegistry()
    registry.register(
        SlashCommand(
            name="one",
            aliases=("o",),
            description="one",
            handler=_noop_handler,
        )
    )

    try:
        registry.register(
            SlashCommand(
                name="two",
                aliases=("o",),
                description="two",
                handler=_noop_handler,
            )
        )
    except ValueError as exc:
        assert "斜杠命令冲突: /o" in str(exc)
    else:
        raise AssertionError("重复别名必须报错")


def test_registry_parses_command_arguments() -> None:
    """注册表能把斜杠输入解析为命令名和参数。"""

    registry = SlashCommandRegistry()

    parsed = registry.parse("/compact focus this")

    assert parsed is not None
    assert parsed.name == "compact"
    assert parsed.args == "focus this"
    assert registry.parse("hello") is None


def test_help_text_is_generated_from_registered_commands() -> None:
    """帮助文本来自注册表中的命令元数据。"""

    registry = create_builtin_command_registry()

    help_text = registry.help_text()
    compact_help = registry.help_text("compact")

    assert "/help [command]" in help_text
    assert "/compact [instructions]" in help_text
    assert "手动压缩当前会话上下文" in compact_help


def test_toolbar_text_shows_matching_commands() -> None:
    """底部命令栏会根据当前斜杠前缀显示候选。"""

    registry = create_builtin_command_registry()

    all_commands = registry.toolbar_text("/")
    help_command = registry.toolbar_text("/he")
    unknown = registry.toolbar_text("/wat")

    assert all_commands is not None
    assert "/help  显示可用命令" in all_commands
    assert "Tab 补全" in all_commands
    assert help_command is not None
    assert "/help  显示可用命令" in help_command
    assert "/compact" not in help_command
    assert unknown == "\ncommands: unknown command · /help"
    assert registry.toolbar_text("hello") is None
    assert registry.toolbar_text("/help compact") is None


def test_complete_text_uses_unique_match_or_common_prefix() -> None:
    """Tab 补全会补唯一命中，或补多候选的公共前缀。"""

    registry = SlashCommandRegistry()
    registry.register(
        SlashCommand(
            name="memory",
            description="memory",
            handler=_noop_handler,
        )
    )
    registry.register(
        SlashCommand(
            name="merge",
            description="merge",
            handler=_noop_handler,
        )
    )
    registry.register(
        SlashCommand(
            name="help",
            description="help",
            handler=_noop_handler,
        )
    )

    assert registry.complete_text("/he") == "/help "
    assert registry.complete_text("/m") == "/me"
    assert registry.complete_text("/me") is None
    assert registry.complete_text("/memory") is None
    assert registry.complete_text("hello") is None


@pytest.mark.asyncio
async def test_builtin_commands_dispatch_locally() -> None:
    """内置命令会调用本地上下文能力，而不是生成普通用户消息。"""

    registry = create_builtin_command_registry()
    context = FakeCommandContext()

    assert await registry.dispatch("/plan", context) == "handled"
    assert context.mode == "plan"

    assert await registry.dispatch("/do", context) == "handled"
    assert context.mode == "default"
    assert context.turns == ["按计划执行。"]

    assert await registry.dispatch("/compact focus", context) == "handled"
    assert context.compactions == ["focus"]

    assert await registry.dispatch("/session", context) == "handled"
    assert context.outputs[-1] == "session status\n"


@pytest.mark.asyncio
async def test_unknown_slash_command_is_handled_with_local_message() -> None:
    """未知斜杠命令会本地提示，不继续作为普通 prompt。"""

    registry = create_builtin_command_registry()
    context = FakeCommandContext()

    outcome = await registry.dispatch("/missing", context)

    assert outcome == "handled"
    assert "未知命令: /missing" in context.outputs[-1]
