"""AgentCode 普通终端界面的公开导出入口。

这里只暴露 terminal-native 应用，不再提供全屏界面组件。
"""

from __future__ import annotations

from agentcode.terminal.app import TerminalApp

__all__ = ["TerminalApp"]
