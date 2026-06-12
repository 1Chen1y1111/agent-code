"""
AgentCode 内置工具系统。

负责定义统一工具协议、执行结果、注册中心，以及默认七个文件/命令工具的装配。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from agentcode.llm import TextContent, ToolDefinition, ToolResultContent

DEFAULT_TIMEOUT = 30.0
ExecutionMode = Literal["parallel", "sequential"]
ToolUpdate = Callable[["ToolResult"], None]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具执行结果，失败也以值返回给上层。"""

    content: list[ToolResultContent]
    is_error: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    terminate: bool = False


@runtime_checkable
class Tool(Protocol):
    """模型可调用工具的统一协议。"""

    def name(self) -> str:
        """返回模型调用时使用的工具名。"""
        ...

    def label(self) -> str:
        """返回 UI 展示用的短标签。"""
        ...

    def description(self) -> str:
        """返回 provider tools schema 中的工具调用说明。"""
        ...

    def prompt_snippet(self) -> str:
        """返回 system prompt 工具列表中的一句话能力摘要。"""
        ...

    def prompt_guidelines(self) -> list[str]:
        """返回 system prompt 中和此工具相关的行为约束。"""
        ...

    def parameters(self) -> dict[str, Any]:
        """返回 provider 可转发给模型的 JSON Schema 参数定义。"""
        ...

    def execution_mode(self) -> ExecutionMode:
        """返回工具执行安全分类，用于 Agent Loop 分批调度。"""
        ...

    def prepare_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """在 schema 校验前修正模型传入的参数形态。"""
        ...

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """执行已解析和校验的参数，并以结构化结果返回。"""
        ...


class BaseTool:
    """内置工具默认行为，避免每个工具重复 label 和参数预处理。"""

    def name(self) -> str:
        """返回模型调用时使用的工具名。"""

        raise NotImplementedError

    def label(self) -> str:
        """默认用工具名作为 UI 标签。"""

        return self.name()

    def prepare_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """默认不改写模型传入的参数对象。"""

        return args

    def prompt_snippet(self) -> str:
        """默认不把工具写入 system prompt 的工具摘要列表。"""

        return ""

    def prompt_guidelines(self) -> list[str]:
        """默认不为工具追加 system prompt 行为约束。"""

        return []


class Registry:
    """集中登记、导出和执行工具。"""

    def __init__(self) -> None:
        """初始化按注册顺序保存工具的注册中心。"""

        self._order: list[str] = []
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具，并拒绝重复的模型可见工具名。"""

        name = tool.name()
        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")
        self._order.append(name)
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        """按模型请求的工具名查找工具实例。"""

        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        """把注册工具转换为 provider 层统一工具定义。"""

        return [
            ToolDefinition(
                name=tool.name(),
                description=tool.description(),
                parameters=tool.parameters(),
                prompt_snippet=tool.prompt_snippet(),
                prompt_guidelines=tuple(tool.prompt_guidelines()),
            )
            for tool in (self._tools[name] for name in self._order)
        ]

    def execution_mode(self, name: str) -> ExecutionMode | None:
        """按工具名返回执行安全分类，未知工具返回 None。"""

        tool = self.get(name)
        return None if tool is None else tool.execution_mode()

    async def execute(
        self,
        tool_call_id: str,
        name: str,
        args: dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """执行指定工具，并把未知工具、参数错误、超时和异常转成结果。"""

        tool = self.get(name)
        if tool is None:
            return text_result(f"未知工具: {name}", is_error=True)

        if not isinstance(args, dict):
            return text_result("工具参数必须是 JSON 对象", is_error=True)

        try:
            prepared = tool.prepare_arguments(args)
        except Exception as exc:  # noqa: BLE001 - 参数修正失败需要回灌给模型。
            return text_result(f"工具 {name} 参数修正失败: {exc}", is_error=True)
        if not isinstance(prepared, dict):
            return text_result("prepare_arguments 必须返回 JSON 对象", is_error=True)

        error = _validate_arguments(tool.parameters(), prepared)
        if error is not None:
            return text_result(error, is_error=True)

        try:
            return await asyncio.wait_for(
                tool.execute(tool_call_id, prepared, on_update=on_update),
                timeout=timeout,
            )
        except TimeoutError:
            return text_result(f"工具 {name} 执行超时（{timeout}s）", is_error=True)
        except Exception as exc:  # noqa: BLE001 - 工具异常必须回灌给模型而非打断会话。
            return text_result(f"工具 {name} 异常: {exc}", is_error=True)


def create_default_registry() -> Registry:
    """创建并注册 AgentCode 当前固定的七个内置工具。"""

    from agentcode.tool.bash import BashTool
    from agentcode.tool.edit_file import EditFileTool
    from agentcode.tool.glob_tool import GlobTool
    from agentcode.tool.grep_tool import GrepTool
    from agentcode.tool.ls_tool import LsTool
    from agentcode.tool.read_file import ReadFileTool
    from agentcode.tool.write_file import WriteFileTool

    registry = Registry()
    registry.register(ReadFileTool())
    registry.register(BashTool())
    registry.register(EditFileTool())
    registry.register(WriteFileTool())
    registry.register(GrepTool())
    registry.register(GlobTool())
    registry.register(LsTool())
    return registry


def text_result(
    text: str,
    is_error: bool = False,
    details: dict[str, Any] | None = None,
    terminate: bool = False,
) -> ToolResult:
    """用单个文本块创建工具结果。"""

    return ToolResult(
        content=[TextContent(text=text)],
        is_error=is_error,
        details={} if details is None else details,
        terminate=terminate,
    )


def content_text(content: list[ToolResultContent]) -> str:
    """把工具结果文本块拼成 provider 和 UI 可展示的纯文本。"""

    return "\n".join(
        block.text for block in content if isinstance(block, TextContent)
    )


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


def _validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """按当前内置工具 schema 做最小必要参数校验。"""

    if schema.get("type") != "object":
        return None

    required = schema.get("required", [])
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in args:
                return f"缺少必填参数: {name}"

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if not _matches_json_type(value, expected):
            expected_text = expected if isinstance(expected, str) else "指定类型"
            return f"参数 {name} 必须是 {expected_text}"
    return None


def _matches_json_type(value: Any, expected: object) -> bool:
    """判断 Python 值是否匹配常见 JSON Schema type。"""

    if expected is None:
        return True
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True
