from __future__ import annotations

import json

import pytest

from agentcode.tool import Registry, new_default_registry
from agentcode.tool.edit_file import EditFileTool
from agentcode.tool.read_file import ReadFileTool
from agentcode.tool.write_file import WriteFileTool


@pytest.mark.asyncio
async def test_registry_exports_default_tools_in_order() -> None:
    registry = new_default_registry()

    definitions = registry.definitions()

    assert [definition.name for definition in definitions] == [
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "glob",
        "grep",
    ]
    assert registry.get("read_file") is not None
    assert registry.get("missing") is None
    result = await registry.execute("missing", "{}")
    assert result.is_error
    assert "未知工具" in result.content


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = Registry()
    tool = ReadFileTool()

    registry.register(tool)

    with pytest.raises(ValueError, match="工具已注册"):
        registry.register(tool)


@pytest.mark.asyncio
async def test_read_file_returns_numbered_content(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")

    result = await ReadFileTool().execute(json.dumps({"path": str(path)}))

    assert not result.is_error
    assert "     1\talpha" in result.content
    assert "     2\tbeta" in result.content


@pytest.mark.asyncio
async def test_read_file_reports_missing_and_directory(tmp_path) -> None:
    missing = await ReadFileTool().execute(
        json.dumps({"path": str(tmp_path / "missing.txt")})
    )
    directory = await ReadFileTool().execute(json.dumps({"path": str(tmp_path)}))

    assert missing.is_error
    assert "文件不存在" in missing.content
    assert directory.is_error
    assert "路径是目录" in directory.content


@pytest.mark.asyncio
async def test_write_file_creates_parent_and_overwrites(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "note.txt"
    tool = WriteFileTool()

    first = await tool.execute(json.dumps({"path": str(path), "content": "one"}))
    second = await tool.execute(json.dumps({"path": str(path), "content": "two"}))

    assert not first.is_error
    assert not second.is_error
    assert path.read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_edit_file_replaces_unique_match(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello old world", encoding="utf-8")

    result = await EditFileTool().execute(
        json.dumps(
            {
                "path": str(path),
                "old_string": "old",
                "new_string": "new",
            }
        )
    )

    assert not result.is_error
    assert path.read_text(encoding="utf-8") == "hello new world"


@pytest.mark.asyncio
async def test_edit_file_reports_zero_and_multiple_matches(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("old and old", encoding="utf-8")
    tool = EditFileTool()

    zero = await tool.execute(
        json.dumps(
            {
                "path": str(path),
                "old_string": "missing",
                "new_string": "new",
            }
        )
    )
    multiple = await tool.execute(
        json.dumps(
            {
                "path": str(path),
                "old_string": "old",
                "new_string": "new",
            }
        )
    )

    assert zero.is_error
    assert zero.content == "未找到匹配的内容"
    assert multiple.is_error
    assert "匹配到 2 处" in multiple.content


@pytest.mark.asyncio
async def test_bash_returns_output_and_timeout() -> None:
    registry = new_default_registry()

    ok = await registry.execute("bash", json.dumps({"command": "echo hi"}))
    timeout = await registry.execute(
        "bash", json.dumps({"command": "sleep 5"}), timeout=0.05
    )

    assert not ok.is_error
    assert "exit_code: 0" in ok.content
    assert "hi" in ok.content
    assert timeout.is_error
    assert "超时" in timeout.content


@pytest.mark.asyncio
async def test_glob_matches_python_files() -> None:
    registry = new_default_registry()

    result = await registry.execute(
        "glob", json.dumps({"pattern": "**/*.py", "path": "src/agentcode"})
    )

    assert not result.is_error
    assert "src/agentcode/cli.py" in result.content


@pytest.mark.asyncio
async def test_grep_matches_known_keyword_and_reports_bad_regex() -> None:
    registry = new_default_registry()

    match = await registry.execute(
        "grep",
        json.dumps(
            {
                "pattern": "AgentCode",
                "path": "src/agentcode",
                "glob": "**/*.py",
            }
        ),
    )
    bad = await registry.execute("grep", json.dumps({"pattern": "["}))

    assert not match.is_error
    assert "AgentCode" in match.content
    assert bad.is_error
    assert "正则非法" in bad.content
