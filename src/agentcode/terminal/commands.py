"""AgentCode 终端斜杠命令注册与分发。

本模块负责命令元数据、冲突检测、帮助文本和 prompt_toolkit 补全；不直接依赖
TerminalApp 的具体实现。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from agentcode.permission import PermissionMode

CommandKind = Literal["local", "mode", "agent"]
CommandOutcome = Literal["handled", "exit"]
MAX_TOOLBAR_COMMANDS = 5


class CommandContext(Protocol):
    """斜杠命令可访问的最小终端能力。"""

    def write(self, text: str, *, style: str | None = None) -> None:
        """向终端追加一段本地命令输出。"""

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """切换当前权限模式。"""

    async def run_turn(self, message: str) -> None:
        """把一条消息提交给 Agent 作为普通用户回合。"""

    async def run_compact(self, custom_instructions: str | None = None) -> None:
        """触发手动上下文压缩。"""

    async def run_resume(self) -> None:
        """打开会话恢复流程。"""

    def session_status(self) -> str:
        """返回当前会话状态文本。"""

    def memory_status(self) -> str:
        """返回当前记忆系统状态文本。"""

    def permission_status(self) -> str:
        """返回当前权限模式状态文本。"""

    def tools_status(self) -> str:
        """返回当前工具状态文本。"""


CommandHandler = Callable[[CommandContext, str], Awaitable[CommandOutcome]]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """一条可由输入框触发的斜杠命令。"""

    name: str
    description: str
    handler: CommandHandler
    usage: str = ""
    aliases: tuple[str, ...] = ()
    kind: CommandKind = "local"


@dataclass(frozen=True, slots=True)
class ParsedSlashCommand:
    """一次用户输入解析出的命令名和参数。"""

    name: str
    args: str


class SlashCommandRegistry:
    """按名称和别名登记斜杠命令，并提供统一分发入口。"""

    def __init__(self) -> None:
        """初始化空注册表。"""

        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}
        self._order: list[str] = []

    def register(self, command: SlashCommand) -> None:
        """注册命令，并在启动期拒绝名字或别名冲突。"""

        name = _normalize_name(command.name)
        aliases = tuple(_normalize_name(alias) for alias in command.aliases)
        if not name:
            raise ValueError("斜杠命令名不能为空")
        if name in self._commands or name in self._aliases:
            raise ValueError(f"斜杠命令冲突: /{name}")
        for alias in aliases:
            if not alias:
                raise ValueError(f"斜杠命令 /{name} 的别名不能为空")
            if alias in self._commands or alias in self._aliases:
                raise ValueError(f"斜杠命令冲突: /{alias}")

        normalized = SlashCommand(
            name=name,
            description=command.description,
            handler=command.handler,
            usage=command.usage,
            aliases=aliases,
            kind=command.kind,
        )
        self._commands[name] = normalized
        self._order.append(name)
        for alias in aliases:
            self._aliases[alias] = name

    def commands(self) -> tuple[SlashCommand, ...]:
        """按注册顺序返回主命令列表。"""

        return tuple(self._commands[name] for name in self._order)

    def get(self, name: str) -> SlashCommand | None:
        """按主名或别名查找命令。"""

        normalized = _normalize_name(name)
        target = self._aliases.get(normalized, normalized)
        return self._commands.get(target)

    def parse(self, text: str) -> ParsedSlashCommand | None:
        """把输入解析成命令调用；非斜杠输入返回 None。"""

        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        body = stripped[1:]
        if not body:
            return ParsedSlashCommand(name="", args="")
        name, separator, args = body.partition(" ")
        return ParsedSlashCommand(name=name, args=args.strip() if separator else "")

    async def dispatch(
        self,
        text: str,
        context: CommandContext,
    ) -> CommandOutcome | None:
        """处理一条输入；非斜杠输入返回 None，未知斜杠命令会本地提示。"""

        parsed = self.parse(text)
        if parsed is None:
            return None
        if not parsed.name:
            context.write(self.help_text(), style="dim")
            return "handled"
        command = self.get(parsed.name)
        if command is None:
            context.write(
                f"未知命令: /{parsed.name}。输入 /help 查看可用命令。\n",
                style="bold red",
            )
            return "handled"
        return await command.handler(context, parsed.args)

    def help_text(self, command_name: str = "") -> str:
        """生成全部命令或单条命令的帮助文本。"""

        if command_name:
            command = self.get(command_name)
            if command is None:
                return f"未知命令: /{_normalize_name(command_name)}。输入 /help 查看可用命令。\n"
            return _command_detail(command)

        lines = ["可用命令："]
        for command in self.commands():
            aliases = (
                f" (别名: {', '.join('/' + alias for alias in command.aliases)})"
                if command.aliases
                else ""
            )
            usage = command.usage or f"/{command.name}"
            lines.append(f"  {usage:<24} {command.description}{aliases}")
        lines.append("输入 /help <命令> 查看详情。")
        return "\n".join(lines) + "\n"

    def completion_items(self, prefix: str) -> tuple[tuple[str, str], ...]:
        """返回与前缀匹配的补全项和值描述。"""

        normalized = _normalize_name(prefix)
        items: list[tuple[str, str]] = []
        for command in self.commands():
            if command.name.startswith(normalized):
                items.append((command.name, command.description))
            for alias in command.aliases:
                if alias.startswith(normalized):
                    items.append((alias, f"{command.description} (/{command.name})"))
        return tuple(items)

    def toolbar_text(self, text: str, *, limit: int = MAX_TOOLBAR_COMMANDS) -> str | None:
        """根据当前输入生成底部命令栏文本，非命令名输入返回 None。"""

        prefix = _command_prefix_from_text(text)
        if prefix is None:
            return None
        items = self.completion_items(prefix)
        if not items:
            return "\ncommands: unknown command · /help"
        rendered = [
            f"/{name}  {description}"
            for name, description in items[: max(1, limit)]
        ]
        more = "" if len(items) <= limit else f"  +{len(items) - limit}"
        return f"\ncommands: {'   '.join(rendered)}{more}   Tab 补全 · Enter 执行"

    def complete_text(self, text: str) -> str | None:
        """按当前命令名前缀计算 Tab 补全后的输入。"""

        prefix = _command_prefix_from_text(text)
        if prefix is None:
            return None
        items = self.completion_items(prefix)
        if not items:
            return None
        names = [name for name, _ in items]
        completed = names[0] if len(names) == 1 else _common_prefix(names)
        if not completed or completed == _normalize_name(prefix):
            return None
        suffix = " " if len(names) == 1 else ""
        return f"/{completed}{suffix}"


def create_builtin_command_registry() -> SlashCommandRegistry:
    """创建 AgentCode 内置斜杠命令注册表。"""

    registry = SlashCommandRegistry()
    registry.register(
        SlashCommand(
            name="help",
            aliases=("?",),
            description="显示可用命令",
            usage="/help [command]",
            handler=lambda ctx, args: _help_handler(ctx, args, registry),
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="exit",
            aliases=("quit",),
            description="退出 AgentCode",
            usage="/exit",
            handler=_exit_handler,
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="plan",
            description="切换到只读计划模式",
            usage="/plan",
            handler=_plan_handler,
            kind="mode",
        )
    )
    registry.register(
        SlashCommand(
            name="do",
            description="退出计划模式并按计划执行",
            usage="/do",
            handler=_do_handler,
            kind="agent",
        )
    )
    registry.register(
        SlashCommand(
            name="compact",
            description="手动压缩当前会话上下文",
            usage="/compact [instructions]",
            handler=_compact_handler,
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="resume",
            description="从历史会话恢复",
            usage="/resume",
            handler=_resume_handler,
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="session",
            description="显示当前会话状态",
            usage="/session",
            handler=_session_handler,
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="memory",
            description="显示记忆系统状态",
            usage="/memory",
            handler=_memory_handler,
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="permissions",
            aliases=("permission",),
            description="显示当前权限模式",
            usage="/permissions",
            handler=_permissions_handler,
            kind="local",
        )
    )
    registry.register(
        SlashCommand(
            name="tools",
            description="显示当前可用工具",
            usage="/tools",
            handler=_tools_handler,
            kind="local",
        )
    )
    return registry


async def _help_handler(
    context: CommandContext,
    args: str,
    registry: SlashCommandRegistry,
) -> CommandOutcome:
    """输出注册表生成的帮助内容。"""

    context.write(registry.help_text(args), style="dim")
    return "handled"


async def _exit_handler(context: CommandContext, args: str) -> CommandOutcome:
    """请求退出主输入循环。"""

    return "exit"


async def _plan_handler(context: CommandContext, args: str) -> CommandOutcome:
    """切换到计划模式。"""

    context.set_permission_mode("plan")
    context.write("已进入 plan 模式。\n", style="dim")
    return "handled"


async def _do_handler(context: CommandContext, args: str) -> CommandOutcome:
    """切回默认权限模式并提交执行提示。"""

    context.set_permission_mode("default")
    await context.run_turn("按计划执行。")
    return "handled"


async def _compact_handler(context: CommandContext, args: str) -> CommandOutcome:
    """执行手动上下文压缩。"""

    await context.run_compact(args or None)
    return "handled"


async def _resume_handler(context: CommandContext, args: str) -> CommandOutcome:
    """执行会话恢复流程。"""

    await context.run_resume()
    return "handled"


async def _session_handler(context: CommandContext, args: str) -> CommandOutcome:
    """输出当前会话状态。"""

    context.write(context.session_status(), style="dim")
    return "handled"


async def _memory_handler(context: CommandContext, args: str) -> CommandOutcome:
    """输出记忆系统状态。"""

    context.write(context.memory_status(), style="dim")
    return "handled"


async def _permissions_handler(context: CommandContext, args: str) -> CommandOutcome:
    """输出权限模式状态。"""

    context.write(context.permission_status(), style="dim")
    return "handled"


async def _tools_handler(context: CommandContext, args: str) -> CommandOutcome:
    """输出当前工具状态。"""

    context.write(context.tools_status(), style="dim")
    return "handled"


def _command_detail(command: SlashCommand) -> str:
    """生成单条命令的详情文本。"""

    lines = [
        f"/{command.name}",
        f"用途: {command.description}",
        f"用法: {command.usage or '/' + command.name}",
        f"类型: {command.kind}",
    ]
    if command.aliases:
        lines.append(f"别名: {', '.join('/' + alias for alias in command.aliases)}")
    return "\n".join(lines) + "\n"


def _normalize_name(name: str) -> str:
    """规范化命令名，允许调用方带或不带前导斜杠。"""

    return name.strip().removeprefix("/").casefold()


def _command_prefix_from_text(text: str) -> str | None:
    """从当前输入提取可补全的命令名前缀。"""

    if not text.startswith("/") or " " in text:
        return None
    return text[1:]


def _common_prefix(values: list[str]) -> str:
    """返回一组候选命令名的公共前缀。"""

    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while not value.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
