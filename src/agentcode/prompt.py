"""
内置提示词、提示模块装配和启动横幅资源。

负责把稳定系统指令、项目上下文和运行时补充指令拆开组织；不直接依赖 LLM 消息类型。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from rich.text import Text

SUPPLEMENTAL_INSTRUCTION_TAG = "agentcode_supplemental_instruction"


@dataclass(frozen=True, slots=True)
class PromptModule:
    """一段稳定系统提示模块，按 priority 从小到大拼装。"""

    id: str
    priority: int
    content: str


@dataclass(frozen=True, slots=True)
class PromptContextFile:
    """已读取的项目级上下文文件，内容会进入稳定系统提示。"""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class PromptSkill:
    """可展示给模型的技能说明，当前只负责进入提示词索引。"""

    name: str
    description: str
    content: str
    file_path: str
    disable_model_invocation: bool = False


@dataclass(frozen=True, slots=True)
class SupplementalInstruction:
    """运行时补充指令，发送给模型但不写入会话历史。"""

    source: str
    content: str
    display: bool = False


@dataclass(frozen=True, slots=True)
class PromptBuildOptions:
    """构建本轮提示上下文需要的输入，字段语义对齐统一 Agent 协议。"""

    custom_prompt: str | None = None
    selected_tools: Sequence[str] | None = None
    tool_snippets: Mapping[str, str] = field(default_factory=dict)
    prompt_guidelines: Sequence[str] = field(default_factory=tuple)
    append_system_prompt: str | None = None
    cwd: str | None = None
    context_files: Sequence[PromptContextFile] = field(default_factory=tuple)
    skills: Sequence[PromptSkill] = field(default_factory=tuple)
    modules: Sequence[PromptModule] | None = None


DEFAULT_TOOL_SNIPPETS: Mapping[str, str] = {
    "read": "读取文件内容",
    "bash": "执行 shell 命令",
    "edit": "用精确文本替换修改文件",
    "write": "创建或完整覆盖文件",
    "grep": "搜索文件内容",
    "find": "按 glob 查找文件路径",
    "ls": "列出目录直接子项",
}


def default_prompt_modules(
    selected_tools: Sequence[str],
    tool_snippets: Mapping[str, str],
    prompt_guidelines: Sequence[str],
) -> list[PromptModule]:
    """生成默认稳定系统提示模块；新增模块不需要改装配函数。"""

    return [
        PromptModule(
            id="identity",
            priority=100,
            content=(
                "You are AgentCode, a concise and helpful terminal AI assistant. "
                "You help users by reading project files, running commands, editing code, "
                "and explaining results clearly."
            ),
        ),
        PromptModule(
            id="tools",
            priority=200,
            content=_tool_section(selected_tools, tool_snippets),
        ),
        PromptModule(
            id="guidelines",
            priority=300,
            content=_guidelines_section(prompt_guidelines),
        ),
    ]


def build_system_prompt(options: PromptBuildOptions | None = None) -> str:
    """把系统提示模块拼成 provider 接收的 system prompt 文本。"""

    opts = options or PromptBuildOptions()
    prompt = opts.custom_prompt.strip() if opts.custom_prompt else ""
    if not prompt:
        selected_tools = list(opts.selected_tools or DEFAULT_TOOL_SNIPPETS.keys())
        tool_snippets = {**DEFAULT_TOOL_SNIPPETS, **dict(opts.tool_snippets)}
        modules = opts.modules or default_prompt_modules(
            selected_tools,
            tool_snippets,
            opts.prompt_guidelines,
        )
        prompt = "\n\n".join(
            module.content.strip()
            for module in sorted(modules, key=lambda module: module.priority)
            if module.content.strip()
        )

    if opts.append_system_prompt and opts.append_system_prompt.strip():
        prompt = f"{prompt}\n\n{opts.append_system_prompt.strip()}"

    project_context = _project_context_section(opts.context_files)
    if project_context:
        prompt = f"{prompt}\n\n{project_context}"

    skills = _skills_section(opts.skills)
    if skills:
        prompt = f"{prompt}\n\n{skills}"

    environment = _environment_section(opts)
    if environment:
        prompt = f"{prompt}\n{environment}"

    return prompt


def runtime_environment_instruction(
    options: PromptBuildOptions,
    today: date | None = None,
) -> SupplementalInstruction | None:
    """生成动态环境补充消息；保留给不进入 system prompt 的运行时上下文。"""

    parts: list[str] = []
    current_date = (today or date.today()).isoformat()
    parts.append(f"Current date: {current_date}")
    if options.cwd:
        parts.append(f"Current working directory: {options.cwd}")
    if not parts:
        return None
    return SupplementalInstruction(source="environment", content="\n".join(parts))


def format_supplemental_instruction(instruction: SupplementalInstruction) -> str:
    """把补充指令包进特殊标签，让模型知道它不是用户提问。"""

    source = _escape_attr(instruction.source)
    display = "true" if instruction.display else "false"
    return (
        f'<{SUPPLEMENTAL_INSTRUCTION_TAG} source="{source}" display="{display}">\n'
        "The following content is supplemental instruction or runtime context. "
        "Do not answer it directly; use it only to guide the next assistant response.\n\n"
        f"{instruction.content.strip()}\n"
        f"</{SUPPLEMENTAL_INSTRUCTION_TAG}>"
    )


PET_BANNER = r"""
  /\___/\
 ( -.-  )
<|  ^  |>
  \___/
""".strip("\n")


def render_banner(version: str, cwd: str) -> Text:
    """生成启动时展示的 ASCII banner 和当前工作目录信息。"""

    # 直接返回 Rich Text，避免开启 markup 后误解析用户内容里的方括号。
    banner = Text()
    banner.append("\n")
    banner.append(f"{PET_BANNER}\n\n", style="bold yellow")
    banner.append("AgentCode ", style="bold cyan")
    banner.append(f"v{version}\n\n", style="bold white")
    banner.append("cwd: ", style="dim")
    banner.append(f"{cwd}\n\n", style="cyan")
    banner.append("Ready. ", style="bold green")
    banner.append("Tools enabled. No MCP.\n\n", style="dim")
    return banner


def _tool_section(
    selected_tools: Sequence[str],
    tool_snippets: Mapping[str, str],
) -> str:
    """生成稳定工具索引，和 provider tools 字段互相强化。"""

    visible = [name for name in selected_tools if tool_snippets.get(name)]
    tools = (
        "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible)
        if visible
        else "(none)"
    )
    return (
        "Available tools:\n"
        f"{tools}\n\n"
        "Prefer the most specific dedicated tool for the job: read files with read, "
        "search content with grep, find paths with find, list directories with ls, "
        "edit existing files with edit, and write new or full files with write."
    )


def _guidelines_section(prompt_guidelines: Sequence[str]) -> str:
    """生成默认行为约束，并去重追加调用方传入的 guideline。"""

    guidelines = [
        "Be concise in final responses and preserve code formatting.",
        "Use tools when you need current project context or need to perform file or command actions.",
        "Before editing an existing file, read the relevant content first so changes match the real file state.",
        "After tool results are provided, continue from the new evidence instead of repeating assumptions.",
        "Show file paths clearly when discussing file changes.",
    ]
    seen = set(guidelines)
    for guideline in prompt_guidelines:
        normalized = guideline.strip()
        if normalized and normalized not in seen:
            guidelines.append(normalized)
            seen.add(normalized)
    return "Guidelines:\n" + "\n".join(f"- {guideline}" for guideline in guidelines)


def _project_context_section(context_files: Sequence[PromptContextFile]) -> str:
    """把项目级指令文件格式化为 XML 风格块，方便模型分辨来源。"""

    if not context_files:
        return ""
    blocks = [
        "<project_context>",
        "Project-specific instructions and guidelines:",
    ]
    for context_file in context_files:
        path = _escape_attr(context_file.path)
        blocks.append(
            f'<project_instructions path="{path}">\n'
            f"{context_file.content.rstrip()}\n"
            "</project_instructions>"
        )
    blocks.append("</project_context>")
    return "\n\n".join(blocks)


def _skills_section(skills: Sequence[PromptSkill]) -> str:
    """格式化可由模型主动选择的技能索引。"""

    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""
    lines = ["<available_skills>"]
    for skill in visible:
        lines.append(
            "<skill>\n"
            f"<name>{_escape_text(skill.name)}</name>\n"
            f"<description>{_escape_text(skill.description)}</description>\n"
            f"<path>{_escape_text(skill.file_path)}</path>\n"
            "</skill>"
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _environment_section(options: PromptBuildOptions) -> str:
    """按统一 system prompt 约定追加当前日期和工作目录。"""

    lines = [f"Current date: {date.today().isoformat()}"]
    if options.cwd:
        lines.append(f"Current working directory: {options.cwd}")
    return "\n".join(lines)


def _escape_attr(value: str) -> str:
    """转义 XML 属性中的特殊字符。"""

    return _escape_text(value).replace('"', "&quot;")


def _escape_text(value: str) -> str:
    """转义 XML 文本中的特殊字符。"""

    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


SYSTEM_PROMPT = build_system_prompt()
