"""
AgentCode 的 MCP 配置解析。

负责从用户级和项目级 YAML 中容错读取 mcp_servers，不处理 provider 配置或网络连接。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys
from typing import Any, Literal, TextIO

import yaml

USER_CONFIG_PATH = Path.home() / ".agentcode" / "config.yaml"
PROJECT_CONFIG_PATH = Path(".agentcode/config.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
McpServerType = Literal["stdio", "http"]


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """单个 MCP server 的启动配置。"""

    name: str
    type: McpServerType
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    trust_annotations: bool = False


def load_mcp_server_configs(
    project_root: str | Path,
    *,
    user_path: str | Path | None = None,
    project_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
    stderr: TextIO | None = None,
) -> tuple[McpServerConfig, ...]:
    """读取并合并用户级与项目级 MCP server 配置。"""

    root = Path(project_root)
    env = dict(os.environ if environ is None else environ)
    err = stderr or sys.stderr
    user_servers = _load_config_file(
        Path(user_path) if user_path else USER_CONFIG_PATH,
        env,
        err,
    )
    project_servers = _load_config_file(
        Path(project_path) if project_path else root / PROJECT_CONFIG_PATH,
        env,
        err,
    )
    merged = dict(user_servers)
    merged.update(project_servers)
    return tuple(merged.values())


def _load_config_file(
    path: Path,
    environ: dict[str, str],
    stderr: TextIO,
) -> dict[str, McpServerConfig]:
    """从单个 YAML 文件读取 mcp_servers，文件问题只返回空集合。"""

    raw = _read_yaml_mapping(path, stderr)
    if raw is None:
        return {}
    servers = raw.get("mcp_servers")
    if servers in (None, {}):
        return {}
    if not isinstance(servers, dict):
        _warn(stderr, f"MCP 配置 {path} 的 mcp_servers 必须是 YAML 对象，已跳过。")
        return {}

    parsed: dict[str, McpServerConfig] = {}
    for raw_name, raw_server in servers.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            _warn(stderr, f"MCP 配置 {path} 中存在非法 server 名，已跳过。")
            continue
        server = _parse_server(raw_name.strip(), raw_server, environ, stderr, path)
        if server is not None:
            parsed[server.name] = server
    return parsed


def _read_yaml_mapping(path: Path, stderr: TextIO) -> dict[str, Any] | None:
    """读取 YAML 对象；文件缺失返回 None，格式问题告警后返回 None。"""

    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _warn(stderr, f"MCP 配置 {path} 无法读取或 YAML 无效，已跳过：{exc}")
        return None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        _warn(stderr, f"MCP 配置 {path} 顶层必须是 YAML 对象，已跳过。")
        return None
    return dict(raw)


def _parse_server(
    name: str,
    raw: Any,
    environ: dict[str, str],
    stderr: TextIO,
    path: Path,
) -> McpServerConfig | None:
    """把单个 server YAML 对象解析成结构化配置，非法时告警跳过。"""

    if not isinstance(raw, dict):
        _warn(stderr, f"MCP server {name} 必须是 YAML 对象，已跳过。")
        return None
    server_type = raw.get("type")
    if server_type not in ("stdio", "http"):
        _warn(stderr, f"MCP server {name} 的 type 必须是 stdio 或 http，已跳过。")
        return None
    trust_annotations = raw.get("trust_annotations", False)
    if not isinstance(trust_annotations, bool):
        _warn(
            stderr,
            f"MCP server {name} 的 trust_annotations 必须是布尔值，已按 false 处理。",
        )
        trust_annotations = False
    if server_type == "stdio":
        return _parse_stdio_server(
            name,
            raw,
            environ,
            stderr,
            trust_annotations,
        )
    return _parse_http_server(
        name,
        raw,
        environ,
        stderr,
        trust_annotations,
        path,
    )


def _parse_stdio_server(
    name: str,
    raw: dict[str, Any],
    environ: dict[str, str],
    stderr: TextIO,
    trust_annotations: bool,
) -> McpServerConfig | None:
    """解析 stdio 类型 server 的 command、args 和 env 字段。"""

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        _warn(stderr, f"MCP stdio server {name} 缺少 command，已跳过。")
        return None
    args = _string_list(raw.get("args"), "args", name, stderr)
    if args is None:
        return None
    env = _string_map(raw.get("env"), "env", name, environ, stderr)
    if env is None:
        return None
    return McpServerConfig(
        name=name,
        type="stdio",
        command=command.strip(),
        args=tuple(args),
        env=env,
        trust_annotations=trust_annotations,
    )


def _parse_http_server(
    name: str,
    raw: dict[str, Any],
    environ: dict[str, str],
    stderr: TextIO,
    trust_annotations: bool,
    path: Path,
) -> McpServerConfig | None:
    """解析 Streamable HTTP 类型 server 的 url 和 headers 字段。"""

    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        _warn(stderr, f"MCP http server {name} 缺少 url，已跳过。")
        return None
    headers = _string_map(raw.get("headers"), "headers", name, environ, stderr)
    if headers is None:
        return None
    if raw.get("env") is not None:
        _warn(stderr, f"MCP http server {name} 的 env 字段不会生效，来源：{path}")
    return McpServerConfig(
        name=name,
        type="http",
        url=url.strip(),
        headers=headers,
        trust_annotations=trust_annotations,
    )


def _string_list(
    raw: Any,
    field_name: str,
    server_name: str,
    stderr: TextIO,
) -> list[str] | None:
    """校验可选字符串数组字段。"""

    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        _warn(
            stderr,
            f"MCP server {server_name} 的 {field_name} 必须是字符串数组，已跳过。",
        )
        return None
    return list(raw)


def _string_map(
    raw: Any,
    field_name: str,
    server_name: str,
    environ: dict[str, str],
    stderr: TextIO,
) -> dict[str, str] | None:
    """校验字符串 map，并展开值中的宿主环境变量引用。"""

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        _warn(
            stderr,
            f"MCP server {server_name} 的 {field_name} 必须是字符串对象，已跳过。",
        )
        return None
    expanded: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            _warn(
                stderr,
                f"MCP server {server_name} 的 {field_name} 只能包含字符串键和值，已跳过。",
            )
            return None
        expanded[key] = _expand_env(value, environ, stderr, server_name, field_name)
    return expanded


def _expand_env(
    value: str,
    environ: dict[str, str],
    stderr: TextIO,
    server_name: str,
    field_name: str,
) -> str:
    """展开单个配置值中的 ${VAR} 引用，缺失变量按空串处理。"""

    def replace(match: re.Match[str]) -> str:
        """把一个环境变量占位符替换成宿主环境变量值。"""

        name = match.group(1)
        if name not in environ:
            _warn(
                stderr,
                f"MCP server {server_name} 的 {field_name} 引用了未定义环境变量 {name}，已按空串处理。",
            )
            return ""
        return environ[name]

    return ENV_PATTERN.sub(replace, value)


def _warn(stderr: TextIO, message: str) -> None:
    """向 stderr 输出统一前缀的 MCP 配置告警。"""

    print(f"[MCP] {message}", file=stderr)
