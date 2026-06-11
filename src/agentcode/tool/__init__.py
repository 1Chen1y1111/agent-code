"""AgentCode 内置工具系统。

负责定义统一工具协议、执行结果、注册中心，以及默认六个文件/命令工具的装配。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentcode.llm import ToolDefinition

DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True, slots=True)
class Result:
    """工具执行结果，失败也以值返回给上层。"""

    content: str
    is_error: bool = False


@runtime_checkable
class Tool(Protocol):
    """模型可调用工具的统一协议。"""

    def name(self) -> str: ...

    def description(self) -> str: ...

    def parameters(self) -> dict[str, Any]: ...

    async def execute(self, args: str) -> Result: ...


class Registry:
    """集中登记、导出和执行工具。"""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.name()
        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")
        self._order.append(name)
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name(),
                description=tool.description(),
                input_schema=tool.parameters(),
            )
            for tool in (self._tools[name] for name in self._order)
        ]

    async def execute(
        self, name: str, args: str, timeout: float = DEFAULT_TIMEOUT
    ) -> Result:
        tool = self.get(name)
        if tool is None:
            return Result(content=f"未知工具: {name}", is_error=True)
        try:
            return await asyncio.wait_for(tool.execute(args), timeout=timeout)
        except TimeoutError:
            return Result(content=f"工具 {name} 执行超时（{timeout}s）", is_error=True)
        except Exception as exc:  # noqa: BLE001 - 工具异常必须回灌给模型而非打断会话。
            return Result(content=f"工具 {name} 异常: {exc}", is_error=True)


def new_default_registry() -> Registry:
    """创建并注册 AgentCode 当前固定的六个内置工具。"""

    from agentcode.tool.bash import BashTool
    from agentcode.tool.edit_file import EditFileTool
    from agentcode.tool.glob_tool import GlobTool
    from agentcode.tool.grep_tool import GrepTool
    from agentcode.tool.read_file import ReadFileTool
    from agentcode.tool.write_file import WriteFileTool

    registry = Registry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(BashTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry


def _truncate(text: str, max_lines: int, max_chars: int) -> str:
    """按行数和字符数限制工具结果体量，并用统一标记告知模型。"""

    truncated = False
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    limited = "\n".join(lines)
    if len(limited) > max_chars:
        limited = limited[:max_chars]
        truncated = True
    if truncated:
        limited = limited.rstrip() + "\n[truncated]"
    return limited


def _load_json_object(args: str) -> tuple[dict[str, Any] | None, str | None]:
    """解析模型传入的 JSON 对象，空参数按空对象处理。"""

    import json

    raw = args or "{}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"参数不是合法 JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, "参数必须是 JSON 对象"
    return value, None
