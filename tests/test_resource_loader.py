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
    assert context_files[0].content == "agents\n"


def test_load_project_context_files_orders_parent_before_child(tmp_path) -> None:
    child = tmp_path / "parent" / "child"
    child.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("root", encoding="utf-8")
    (child / "AGENTS.md").write_text("child", encoding="utf-8")

    context_files = load_project_context_files(child)

    assert [context_file.content for context_file in context_files] == [
        "root\n",
        "child\n",
    ]


def test_load_project_context_files_loads_user_before_project(tmp_path) -> None:
    user_dir = tmp_path / "user"
    project = tmp_path / "project"
    user_dir.mkdir()
    project.mkdir()
    (user_dir / "AGENTS.md").write_text("user", encoding="utf-8")
    (project / "AGENTS.md").write_text("project", encoding="utf-8")

    context_files = load_project_context_files(project, user_dir=user_dir)

    assert [context_file.content for context_file in context_files] == [
        "user\n",
        "project\n",
    ]


def test_agents_include_expands_and_detects_cycle(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "root\n@include child.md\n",
        encoding="utf-8",
    )
    (tmp_path / "child.md").write_text(
        "child\n@include AGENTS.md\n",
        encoding="utf-8",
    )

    result = load_prompt_resources(tmp_path, user_dir=tmp_path / "missing-user")

    assert result.prompt_options.context_files[0].content == "root\nchild\n"
    assert len(result.diagnostics) == 1
    assert "环路" in result.diagnostics[0].message


def test_load_prompt_resources_reports_unreadable_context_file(tmp_path) -> None:
    (tmp_path / "AGENTS.md").mkdir()

    result = load_prompt_resources(tmp_path)

    assert result.prompt_options.context_files == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].path == str((tmp_path / "AGENTS.md").resolve())
    assert "无法读取提示上下文文件" in result.diagnostics[0].message
