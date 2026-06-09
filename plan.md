# 多协议 LLM 终端对话客户端 Plan

## 技术栈

- 语言：Python 3.14（沿用当前项目要求 `>=3.14`）。
- TUI：Textual + Rich + Textual CSS。
- 配置：PyYAML（import 名 `yaml`）。
- LLM 通信：官方 Python SDK：`anthropic.AsyncAnthropic`、`openai.AsyncOpenAI`，均使用 async 流式能力。

## 架构概览

当前项目将从 Typer/Rich REPL 迁移为 Textual 全屏 TUI。命名统一为：

- 应用名：AgentCode
- 命令：`agentcode`
- Python 包：`agentcode`
- 源码目录：`src/agentcode/`
- 配置路径：`.agentcode/config.yaml`

分层：

1. 入口层 `agentcode.cli`：加载配置、启动 Textual App。
2. 配置层 `agentcode.config`：读取并校验 `.agentcode/config.yaml`，产出 providers 列表。
3. LLM 协议层 `agentcode.llm`：定义协议无关 `Provider` Protocol、统一消息和流式事件类型；Anthropic/OpenAI 适配器封装官方 SDK，丢弃 thinking 增量。
4. 会话层 `agentcode.conversation`：进程内维护多轮历史。
5. 提示词层 `agentcode.prompt`：内置 system prompt、ASCII 猫 banner。
6. 终端层 `agentcode.tui`：Textual App，含 provider 选择、对话区、输入框、状态栏、流式显示、markdown 定型和计时。

## 数据流

```text
.agentcode/config.yaml
  → config.load()
  → list[ProviderConfig]
  → 单 provider 直进 / 多 provider OptionList 选择
  → new_provider(选定配置)
  → 用户输入
  → Conversation 追加 user
  → Provider.stream(完整历史 + system prompt)
  → StreamEvent(text/done/err)
  → TUI 实时显示文本增量
  → done 后 markdown 定型
  → Conversation 追加 assistant
```

## 核心数据结构

### ProviderConfig

```python
@dataclass
class ProviderConfig:
    name: str
    protocol: Literal["anthropic", "openai"]
    api_key: str
    model: str
    base_url: str | None = None
    thinking: bool = False
```

### Config

```python
@dataclass
class Config:
    providers: list[ProviderConfig]
```

### Message

```python
@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
```

### StreamEvent

```python
@dataclass
class StreamEvent:
    text: str = ""
    done: bool = False
    err: Exception | None = None
```

### Provider

```python
class Provider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]: ...
```

### Conversation

```python
class Conversation:
    def add_user(self, text: str) -> None: ...
    def add_assistant(self, text: str) -> None: ...
    def messages(self) -> list[Message]: ...
```

### SessionState

```python
class SessionState(Enum):
    SELECTING = "selecting"
    IDLE = "idle"
    STREAMING = "streaming"
```

## 模块设计

### `agentcode.config`

职责：读取并校验 `.agentcode/config.yaml`。

接口：

```python
def load(path: str | Path = ".agentcode/config.yaml") -> Config: ...
```

校验：

- `providers` 必须存在且非空。
- 每项 `name` / `protocol` / `api_key` / `model` 必须非空。
- `protocol` 只能是 `anthropic` 或 `openai`。
- `base_url` 可空；空时使用 SDK 默认端点。
- `thinking` 可空；默认 false。
- 错误抛 `ConfigError`，信息指明 provider 索引和字段，且不包含 api key 明文。

### `agentcode.llm`

职责：定义协议无关接口与事件类型，按 protocol 构造 provider。

Anthropic 适配器：

- 使用 `AsyncAnthropic(api_key=..., base_url=...)`。
- 请求携带 `system=SYSTEM_PROMPT` 和完整历史。
- `thinking=True` 时加入 `thinking={"type": "enabled", "budget_tokens": 2048}`。
- 解析正文 text delta 为 `StreamEvent(text=...)`。
- thinking delta 接收后丢弃。
- 正常结束 yield `StreamEvent(done=True)`。
- SDK 异常 yield `StreamEvent(err=exc)`。

OpenAI 适配器：

- 使用 `AsyncOpenAI(api_key=..., base_url=...)`。
- messages 首条插入 system prompt。
- 使用流式 chat completions。
- 解析正文 delta 为 `StreamEvent(text=...)`。
- `thinking` 字段忽略。
- 正常结束 yield `StreamEvent(done=True)`。
- SDK 异常 yield `StreamEvent(err=exc)`。

### `agentcode.conversation`

职责：维护单进程、单会话多轮历史。

接口：

- `add_user(text)`
- `add_assistant(text)`
- `messages()` 返回副本。

### `agentcode.prompt`

职责：提供：

- `SYSTEM_PROMPT`
- `CAT_BANNER`
- `render_banner(version, cwd)`

### `agentcode.tui`

职责：Textual 全屏界面。

主要 widget：

- `RichLog`：对话区，追加完成消息、markdown 回复和错误。
- `Static`：动态流式区和计时提示。
- `TextArea`：底部输入框，多行输入。
- `Static`：底部状态栏。
- `OptionList`：多 provider 启动选择。

状态：

- `SELECTING`：显示 provider 列表，方向键选择，Enter 确认。
- `IDLE`：接受输入。
- `STREAMING`：不接受新提交，异步消费 provider 流，UI 保持响应。

提交流程：

```text
Enter 提交
  → 若文本为 /exit：退出
  → Conversation.add_user
  → RichLog 追加用户块
  → 清空输入框
  → 启动计时器
  → asyncio task 消费 Provider.stream(...)
  → text 增量更新动态区
  → done 后 Markdown 定型写入 RichLog
  → Conversation.add_assistant
  → 回 IDLE
```

错误流程：

```text
Provider 返回 err
  → RichLog 写入红色错误块
  → 停止计时
  → 回 IDLE
  → 不退出
```

退出流程：

```text
/exit 或 Ctrl+C
  → 若存在流式任务则 cancel
  → App.exit()
  → Textual 还原终端状态
```

## 文件组织

```text
src/
└── agentcode/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── config.py
    ├── prompt.py
    ├── conversation.py
    ├── llm/
    │   ├── __init__.py
    │   ├── anthropic_provider.py
    │   └── openai_provider.py
    └── tui/
        ├── __init__.py
        ├── app.py
        └── view.py

.agentcode/
└── config.yaml.example

tests/
├── test_config.py
├── test_conversation.py
├── test_llm.py
└── test_tui.py
```

现有 `agent_code/` 将被 `src/agentcode/` 替代；现有 REPL 相关模块不再保留。

## 技术决策

| 决策点 | 选择 | 理由 |
| --- | --- | --- |
| 应用名 | AgentCode | 按最新命名要求统一为 AgentCode。 |
| 命令 | `agentcode` | 与当前用户确认一致。 |
| 包名 | `agentcode` | 全量改名，避免 `agent_code` 与新规格不一致。 |
| 源码布局 | `src/agentcode` | 匹配新计划，隔离旧 REPL 实现。 |
| 配置路径 | `.agentcode/config.yaml` | 项目内 YAML，且密钥文件可 gitignore。 |
| TUI 框架 | Textual | async-first，适合流式 UI 和全屏终端布局。 |
| 对话区 | `RichLog` | 适合追加历史和滚动回看。 |
| 输入框 | `TextArea` | 支持多行输入，满足 Alt+Enter 换行需求。 |
| Markdown 定型 | Rich Markdown | 回复结束后完整渲染，减少流式中 markdown 抖动。 |
| 协议抽象 | Provider Protocol | 统一 Anthropic/OpenAI 上层行为。 |
| 历史 | 进程内 list | 满足单次会话记忆，不做持久化。 |
