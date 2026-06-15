"""
AgentCode 的提示资源加载器。

负责发现并读取用户和项目上下文文件，把它们转换为 PromptBuildOptions；处理
AGENTS.md 的 include 展开，但不处理 provider、工具协议或 skills。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from agentcode.prompt import PromptBuildOptions, PromptContextFile

CONTEXT_FILE_CANDIDATES = (
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
)
MAX_INCLUDE_DEPTH = 5
INCLUDE_RE = re.compile(r"^\s*@include\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    """提示资源加载时的非致命诊断信息。"""

    path: str
    message: str


@dataclass(frozen=True, slots=True)
class PromptResourceLoadResult:
    """启动期加载到的提示资源和诊断信息。"""

    prompt_options: PromptBuildOptions
    diagnostics: tuple[ResourceDiagnostic, ...] = ()


def load_prompt_resources(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> PromptResourceLoadResult:
    """加载当前工作目录可见的提示资源，并返回可直接传给 Agent 的选项。"""

    context_files, diagnostics = _load_project_context_files_with_diagnostics(
        cwd,
        user_dir=user_dir,
    )
    return PromptResourceLoadResult(
        prompt_options=PromptBuildOptions(context_files=tuple(context_files)),
        diagnostics=tuple(diagnostics),
    )


def load_project_context_files(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> list[PromptContextFile]:
    """加载用户级和项目级上下文文件，按用户、父目录、子目录顺序返回。"""

    context_files, _ = _load_project_context_files_with_diagnostics(
        cwd,
        user_dir=user_dir,
    )
    return context_files


def _load_project_context_files_with_diagnostics(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> tuple[list[PromptContextFile], list[ResourceDiagnostic]]:
    """加载上下文文件并保留不可读文件诊断，供启动期观察。"""

    current = Path(cwd).resolve()
    if not current.is_dir():
        current = current.parent

    ordered_files: list[PromptContextFile] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen_paths: set[Path] = set()
    root = current.anchor

    user_context = _load_context_file_from_dir(
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else Path.home() / ".agentcode",
        diagnostics,
    )
    if user_context is not None:
        seen_paths.add(Path(user_context.path))

    while True:
        context_file = _load_context_file_from_dir(current, diagnostics)
        if context_file is not None:
            path = Path(context_file.path)
            if path not in seen_paths:
                ordered_files.insert(0, context_file)
                seen_paths.add(path)

        if str(current) == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    if user_context is not None:
        ordered_files.insert(0, user_context)
    return ordered_files, diagnostics


def _load_context_file_from_dir(
    directory: Path,
    diagnostics: list[ResourceDiagnostic],
) -> PromptContextFile | None:
    """在单个目录按候选顺序读取第一个可用上下文文件。"""

    for filename in CONTEXT_FILE_CANDIDATES:
        candidate = directory / filename
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            content = _read_context_file_with_includes(
                resolved,
                diagnostics,
                stack=(),
                depth=0,
            )
            return (
                PromptContextFile(
                    path=str(resolved),
                    content=content,
                )
            )
        except OSError as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    path=str(resolved),
                    message=f"无法读取提示上下文文件: {exc}",
                ),
            )
            return None
    return None


def _read_context_file_with_includes(
    path: Path,
    diagnostics: list[ResourceDiagnostic],
    *,
    stack: tuple[Path, ...],
    depth: int,
) -> str:
    """读取上下文文件并递归展开 @include 指令。"""

    resolved = path.resolve()
    if resolved in stack:
        diagnostics.append(
            ResourceDiagnostic(
                path=str(resolved),
                message="AGENTS include 出现环路，已跳过该引用",
            )
        )
        return ""
    if depth > MAX_INCLUDE_DEPTH:
        diagnostics.append(
            ResourceDiagnostic(
                path=str(resolved),
                message=f"AGENTS include 超过最大深度 {MAX_INCLUDE_DEPTH}，已跳过",
            )
        )
        return ""

    lines: list[str] = []
    next_stack = (*stack, resolved)
    for line in resolved.read_text(encoding="utf-8").splitlines():
        match = INCLUDE_RE.match(line)
        if match is None:
            lines.append(line)
            continue
        include_path = _include_path(resolved.parent, match.group(1))
        try:
            lines.append(
                _read_context_file_with_includes(
                    include_path,
                    diagnostics,
                    stack=next_stack,
                    depth=depth + 1,
                ).rstrip()
            )
        except OSError as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    path=str(include_path),
                    message=f"无法读取 AGENTS include 文件: {exc}",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def _include_path(base_dir: Path, raw_value: str) -> Path:
    """把 @include 参数解析为相对当前文件的路径。"""

    text = raw_value.strip().strip('"').strip("'")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()
