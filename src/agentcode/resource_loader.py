"""
AgentCode 的提示资源加载器。

负责发现并读取项目上下文文件，把它们转换为 PromptBuildOptions；不处理 provider、
工具协议、skills 或自定义 system prompt。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentcode.prompt import PromptBuildOptions, PromptContextFile

CONTEXT_FILE_CANDIDATES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


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


def load_prompt_resources(cwd: str | Path) -> PromptResourceLoadResult:
    """加载当前工作目录可见的提示资源，并返回可直接传给 Agent 的选项。"""

    context_files, diagnostics = _load_project_context_files_with_diagnostics(cwd)
    return PromptResourceLoadResult(
        prompt_options=PromptBuildOptions(context_files=tuple(context_files)),
        diagnostics=tuple(diagnostics),
    )


def load_project_context_files(cwd: str | Path) -> list[PromptContextFile]:
    """从 cwd 向上查找项目上下文文件，按父目录到子目录顺序返回。"""

    context_files, _ = _load_project_context_files_with_diagnostics(cwd)
    return context_files


def _load_project_context_files_with_diagnostics(
    cwd: str | Path,
) -> tuple[list[PromptContextFile], list[ResourceDiagnostic]]:
    """加载上下文文件并保留不可读文件诊断，供启动期观察。"""

    current = Path(cwd).resolve()
    if not current.is_dir():
        current = current.parent

    ordered_files: list[PromptContextFile] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen_paths: set[Path] = set()
    root = current.anchor

    while True:
        context_file, diagnostic = _load_context_file_from_dir(current)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
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

    return ordered_files, diagnostics


def _load_context_file_from_dir(
    directory: Path,
) -> tuple[PromptContextFile | None, ResourceDiagnostic | None]:
    """在单个目录按候选顺序读取第一个可用上下文文件。"""

    for filename in CONTEXT_FILE_CANDIDATES:
        candidate = directory / filename
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            return (
                PromptContextFile(
                    path=str(resolved),
                    content=resolved.read_text(encoding="utf-8"),
                ),
                None,
            )
        except OSError as exc:
            return (
                None,
                ResourceDiagnostic(
                    path=str(resolved),
                    message=f"无法读取提示上下文文件: {exc}",
                ),
            )
    return None, None
