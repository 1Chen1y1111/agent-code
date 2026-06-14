"""
AgentCode 的 MCP 客户端集成包。

对外暴露配置加载和生命周期管理入口，隐藏具体 SDK 传输和工具适配细节。
"""

from __future__ import annotations

from agentcode.mcp.config import McpServerConfig, load_mcp_server_configs
from agentcode.mcp.manager import McpManager

__all__ = ["McpManager", "McpServerConfig", "load_mcp_server_configs"]
