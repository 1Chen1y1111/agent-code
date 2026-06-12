from __future__ import annotations

import pytest

from agentcode.tool import (
    BaseTool,
    ExecutionMode,
    Registry,
    ToolResult,
    ToolUpdate,
    content_text,
    create_default_registry,
    text_result,
)
from agentcode.tool.edit_file import EditFileTool
from agentcode.tool.read_file import ReadFileTool
from agentcode.tool.write_file import WriteFileTool


@pytest.mark.asyncio
async def test_registry_exports_default_tools_in_order() -> None:
    registry = create_default_registry()

    definitions = registry.definitions()

    assert [definition.name for definition in definitions] == [
        "read",
        "bash",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    ]
    assert registry.get("read") is not None
    assert registry.get("missing") is None
    result = await registry.execute("call_missing", "missing", {})
    assert result.is_error
    assert "未知工具" in content_text(result.content)


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

    result = await ReadFileTool().execute("call_1", {"path": str(path)})

    assert not result.is_error
    assert "     1\talpha" in content_text(result.content)
    assert "     2\tbeta" in content_text(result.content)


@pytest.mark.asyncio
async def test_read_file_reports_missing_and_directory(tmp_path) -> None:
    missing = await ReadFileTool().execute(
        "call_1", {"path": str(tmp_path / "missing.txt")}
    )
    directory = await ReadFileTool().execute("call_2", {"path": str(tmp_path)})

    assert missing.is_error
    assert "文件不存在" in content_text(missing.content)
    assert directory.is_error
    assert "路径是目录" in content_text(directory.content)


@pytest.mark.asyncio
async def test_write_file_creates_parent_and_overwrites(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "note.txt"
    tool = WriteFileTool()

    first = await tool.execute("call_1", {"path": str(path), "content": "one"})
    second = await tool.execute("call_2", {"path": str(path), "content": "two"})

    assert not first.is_error
    assert not second.is_error
    assert path.read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_edit_file_replaces_unique_match(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello old world", encoding="utf-8")

    result = await EditFileTool().execute(
        "call_1",
        {
            "path": str(path),
            "old_string": "old",
            "new_string": "new",
        },
    )

    assert not result.is_error
    assert path.read_text(encoding="utf-8") == "hello new world"


@pytest.mark.asyncio
async def test_edit_file_reports_zero_and_multiple_matches(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("old and old", encoding="utf-8")
    tool = EditFileTool()

    zero = await tool.execute(
        "call_1",
        {
            "path": str(path),
            "old_string": "missing",
            "new_string": "new",
        },
    )
    multiple = await tool.execute(
        "call_2",
        {
            "path": str(path),
            "old_string": "old",
            "new_string": "new",
        },
    )

    assert zero.is_error
    assert content_text(zero.content) == "未找到匹配的内容"
    assert multiple.is_error
    assert "匹配到 2 处" in content_text(multiple.content)


@pytest.mark.asyncio
async def test_bash_returns_output_and_timeout() -> None:
    registry = create_default_registry()

    ok = await registry.execute("call_1", "bash", {"command": "echo hi"})
    timeout = await registry.execute(
        "call_2", "bash", {"command": "sleep 5"}, timeout=0.05
    )

    assert not ok.is_error
    assert "exit_code: 0" in content_text(ok.content)
    assert "hi" in content_text(ok.content)
    assert timeout.is_error
    assert "超时" in content_text(timeout.content)


@pytest.mark.asyncio
async def test_glob_matches_python_files() -> None:
    registry = create_default_registry()

    result = await registry.execute(
        "call_1",
        "find",
        {"pattern": "**/*.py", "path": "src/agentcode"},
    )

    assert not result.is_error
    assert "src/agentcode/cli.py" in content_text(result.content)


@pytest.mark.asyncio
async def test_grep_matches_known_keyword_and_reports_bad_regex() -> None:
    registry = create_default_registry()

    match = await registry.execute(
        "call_1",
        "grep",
        {
            "pattern": "AgentCode",
            "path": "src/agentcode",
            "glob": "**/*.py",
        },
    )
    bad = await registry.execute("call_2", "grep", {"pattern": "["})

    assert not match.is_error
    assert "AgentCode" in content_text(match.content)
    assert bad.is_error
    assert "正则非法" in content_text(bad.content)


@pytest.mark.asyncio
async def test_registry_validates_and_prepares_arguments() -> None:
    registry = Registry()
    tool = PreparingTool()
    registry.register(tool)

    prepared = await registry.execute("call_1", "prepare", {"file": "note.txt"})
    invalid_args = await registry.execute("call_2", "prepare", "{")  # type: ignore[arg-type]
    missing = await registry.execute("call_3", "prepare", {})
    wrong_type = await registry.execute("call_4", "prepare", {"path": 1})

    assert content_text(prepared.content) == "NOTE.TXT"
    assert invalid_args.is_error
    assert "工具参数必须是 JSON 对象" in content_text(invalid_args.content)
    assert missing.is_error
    assert "缺少必填参数: path" in content_text(missing.content)
    assert wrong_type.is_error
    assert "参数 path 必须是 string" in content_text(wrong_type.content)


@pytest.mark.asyncio
async def test_ls_lists_direct_children(tmp_path) -> None:
    registry = create_default_registry()
    (tmp_path / "dir").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    result = await registry.execute(
        "call_1", "ls", {"path": str(tmp_path)}
    )

    assert not result.is_error
    assert "dir/" in content_text(result.content)
    assert "file.txt" in content_text(result.content)


class PreparingTool(BaseTool):
    def name(self) -> str:
        """返回测试工具名。"""

        return "prepare"

    def description(self) -> str:
        """返回测试工具说明。"""

        return "fake"

    def parameters(self) -> dict[str, object]:
        """声明测试工具需要 path 参数。"""

        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def execution_mode(self) -> ExecutionMode:
        """测试工具允许并发。"""

        return "parallel"

    def prepare_arguments(self, args: dict[str, object]) -> dict[str, object]:
        """把 alias 修正成正式参数名。"""

        if "path" not in args and "file" in args:
            return {"path": args["file"]}
        return args

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, object],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """返回参数值以验证 Registry 已完成 prepare 和校验。"""

        return text_result(str(args["path"]).upper())
