from __future__ import annotations

import io
from pathlib import Path

from agentcode.mcp.config import load_mcp_server_configs


def test_mcp_config_merges_user_and_project_by_server_name(tmp_path: Path) -> None:
    """项目级同名 MCP server 会完整覆盖用户级定义。"""

    user = tmp_path / "user.yaml"
    project = tmp_path / "project.yaml"
    user.write_text(
        """
mcp_servers:
  github:
    type: stdio
    command: old
    args: ["one"]
  sqlite:
    type: http
    url: http://user.example/mcp
""".strip()
        + "\n",
        encoding="utf-8",
    )
    project.write_text(
        """
mcp_servers:
  github:
    type: http
    url: http://project.example/mcp
    headers:
      Authorization: Bearer ${TOKEN}
    trust_annotations: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    servers = load_mcp_server_configs(
        tmp_path,
        user_path=user,
        project_path=project,
        environ={"TOKEN": "secret"},
    )

    assert [server.name for server in servers] == ["github", "sqlite"]
    assert servers[0].type == "http"
    assert servers[0].url == "http://project.example/mcp"
    assert servers[0].headers == {"Authorization": "Bearer secret"}
    assert servers[0].trust_annotations is True
    assert servers[1].url == "http://user.example/mcp"


def test_mcp_config_warns_and_skips_invalid_entries(tmp_path: Path) -> None:
    """非法配置文件或 server 只影响 MCP 加载，不抛异常。"""

    user = tmp_path / "user.yaml"
    project = tmp_path / "project.yaml"
    user.write_text("mcp_servers: [", encoding="utf-8")
    project.write_text(
        """
mcp_servers:
  missing_command:
    type: stdio
  bad_type:
    type: websocket
  good:
    type: stdio
    command: uvx
    args: ["pkg"]
    env:
      API_KEY: ${MISSING}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    stderr = io.StringIO()

    servers = load_mcp_server_configs(
        tmp_path,
        user_path=user,
        project_path=project,
        environ={},
        stderr=stderr,
    )

    assert [server.name for server in servers] == ["good"]
    assert servers[0].env == {"API_KEY": ""}
    rendered = stderr.getvalue()
    assert "YAML 无效" in rendered
    assert "缺少 command" in rendered
    assert "type 必须是 stdio 或 http" in rendered
    assert "未定义环境变量 MISSING" in rendered
