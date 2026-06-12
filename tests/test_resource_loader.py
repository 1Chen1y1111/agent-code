from __future__ import annotations

from agentcode.resource_loader import (
    load_project_context_files,
    load_prompt_resources,
)


def test_load_project_context_files_returns_empty_without_context_files(tmp_path) -> None:
    child = tmp_path / "project"
    child.mkdir()

    context_files = load_project_context_files(child)

    assert context_files == []


def test_load_project_context_files_prefers_agents_over_claude(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("claude", encoding="utf-8")

    context_files = load_project_context_files(tmp_path)

    assert len(context_files) == 1
    assert context_files[0].path == str((tmp_path / "AGENTS.md").resolve())
    assert context_files[0].content == "agents"


def test_load_project_context_files_orders_parent_before_child(tmp_path) -> None:
    child = tmp_path / "parent" / "child"
    child.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("root", encoding="utf-8")
    (child / "AGENTS.md").write_text("child", encoding="utf-8")

    context_files = load_project_context_files(child)

    assert [context_file.content for context_file in context_files] == [
        "root",
        "child",
    ]


def test_load_prompt_resources_reports_unreadable_context_file(tmp_path) -> None:
    (tmp_path / "AGENTS.md").mkdir()

    result = load_prompt_resources(tmp_path)

    assert result.prompt_options.context_files == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].path == str((tmp_path / "AGENTS.md").resolve())
    assert "无法读取提示上下文文件" in result.diagnostics[0].message
