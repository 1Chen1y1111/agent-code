AGENTS.md

````bash
# AgentCode

我正在构建一个终端 AI 编程助手（类似 Claude Code），项目名叫AgentCode，使用 Python 实现。

## 语言
中文回答，中文注释。

## 代码注释与文件职责

- 每个源码文件顶部必须写一句到两句“文件主要职责”，说明这个文件负责什么、边界是什么。
- Python 文件使用模块级 docstring，放在文件最顶部，位于 `from __future__ import ...` 之前。
- 每个函数和方法都必须写简短中文 docstring，包括类方法、属性方法、事件处理方法和私有辅助函数。
- 方法 docstring 重点说明这个方法在流程中的职责、调用边界、输入输出语义或容易误解的状态变化；简单 getter 也要用一句话说明其对外暴露的含义。
- 注释统一使用中文。
- 注释重点解释设计意图、边界条件、协议差异、异步/状态机逻辑、终端兼容、IME/光标等容易踩坑的地方。
- 不要给显而易见的赋值、简单 getter、普通导入写注释。
- 不写作者、日期、变更记录、版权头，避免过期维护成本。
- 修改文件职责时，同步更新文件顶部职责说明。

示例：

```python
"""
AgentCode 的普通终端 CLI 应用。

负责 provider 选择、prompt_toolkit 输入循环，以及把 Session 事件追加渲染到终端 scrollback。
"""

from __future__ import annotations
````

推荐注释：

```python
# Anthropic SDK 会同时产生 raw delta 和 helper text 事件；只吃 text，
# 否则同一个 token 会被追加两次。
```

避免注释：

```python
# 把 text 加到 cur_reply 上
self.cur_reply += text
```

## 测试

开发完功能后，用 tmux 做端到端测试：

1. 在 tmux 中启动 AgentCode
2. 输入一段真实的对话请求
3. 观察 AgentCode 是否正确调用工具、生成回复
4. 对照 checklist.md 逐项验收

```

```

- 叫我靓仔
