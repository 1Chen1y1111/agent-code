# 多协议 LLM 终端对话客户端 Tasks

## 文件清单

| 操作 | 文件 | 职责 |
| --- | --- | --- |
| 修改 | `pyproject.toml` | 依赖、包名、脚本入口、src 布局 |
| 修改 | `uv.lock` | 同步依赖锁 |
| 新建 | `.agentcode/config.yaml.example` | providers 列表配置模板 |
| 修改 | `.gitignore` | 忽略 `.agentcode/config.yaml` |
| 新建 | `src/agentcode/__init__.py` | 包标识、版本号 |
| 新建 | `src/agentcode/__main__.py` | `python -m agentcode` |
| 新建 | `src/agentcode/cli.py` | 加载配置并启动 TUI |
| 新建 | `src/agentcode/config.py` | 配置 dataclass、load、校验 |
| 新建 | `src/agentcode/prompt.py` | system prompt、猫 banner |
| 新建 | `src/agentcode/conversation.py` | 多轮历史 |
| 新建 | `src/agentcode/llm/__init__.py` | Message、StreamEvent、Provider、工厂 |
| 新建 | `src/agentcode/llm/anthropic_provider.py` | Anthropic 适配 |
| 新建 | `src/agentcode/llm/openai_provider.py` | OpenAI 适配 |
| 新建 | `src/agentcode/tui/__init__.py` | TUI 包 |
| 新建 | `src/agentcode/tui/app.py` | Textual App、状态机 |
| 新建 | `src/agentcode/tui/view.py` | 渲染辅助 |
| 删除 | `agent_code/` | 移除旧 REPL 实现 |
| 新建/修改 | `tests/` | 配置、conversation、llm、tui 测试 |
| 修改 | `README.md` | 新配置格式与运行说明 |
| 修改 | `spec.md`、`plan.md`、`task.md`、`checklist.md` | 新 TUI 版本文档 |

## T1: 更新项目元数据和依赖

**文件：** `pyproject.toml`、`uv.lock`
**依赖：** 无
**步骤：**

1. 将项目包配置改为 `src/agentcode` 布局。
2. 保持命令入口 `agentcode = "agentcode.cli:main"`。
3. 添加 `textual` 依赖。
4. 保留 `rich`、`anthropic`、`openai`、`pyyaml`。
5. 使用 `uv lock` / `uv sync` 同步锁文件。

**验证：** `uv run python -c "import textual, rich, anthropic, openai, yaml"` 退出码为 0。

## T2: 建立 agentcode 包骨架

**文件：** `src/agentcode/__init__.py`、`src/agentcode/__main__.py`、`src/agentcode/cli.py`
**依赖：** T1
**步骤：**

1. 定义 `__version__ = "0.1.0"`。
2. `__main__.py` 转调 `cli.main()`。
3. `cli.py` 暂时加载配置并启动后续 TUI；在 TUI 未完成前可先打印版本。

**验证：** `uv run python -m agentcode --help` 或 `uv run agentcode --help` 可执行。

## T3: 实现配置层

**文件：** `src/agentcode/config.py`、`tests/test_config.py`
**依赖：** T2
**步骤：**

1. 定义 `ProviderConfig`、`Config`、`ConfigError`。
2. 实现 `load(path=".agentcode/config.yaml") -> Config`。
3. 读取 YAML，校验 `providers` 非空。
4. 逐项校验 `name`、`protocol`、`api_key`、`model` 非空。
5. 校验 `protocol` 只能为 `anthropic` 或 `openai`。
6. 支持可选 `base_url` 和 `thinking`，`thinking` 默认 false。
7. 错误信息指明 provider 索引和字段，不包含 api key 明文。
8. 添加单元测试覆盖合法配置、缺字段、非法协议、缺文件、YAML 格式错误。

**验证：** `uv run python -m unittest tests.test_config` 通过。

## T4: 添加配置模板和忽略规则

**文件：** `.agentcode/config.yaml.example`、`.gitignore`
**依赖：** T3
**步骤：**

1. 创建 `.agentcode/config.yaml.example`，包含 Anthropic 和 OpenAI providers 示例。
2. `.gitignore` 忽略 `.agentcode/config.yaml`。
3. README 说明复制 example 为真实配置。

**验证：** 复制 example 为 `.agentcode/config.yaml` 后配置测试可读取；`git status --ignored` 显示真实配置被忽略。

## T5: 实现 prompt 和 conversation

**文件：** `src/agentcode/prompt.py`、`src/agentcode/conversation.py`、`tests/test_conversation.py`
**依赖：** T2
**步骤：**

1. 定义 `SYSTEM_PROMPT`。
2. 定义 ASCII 猫 `CAT_BANNER`。
3. 实现 `render_banner(version, cwd)`。
4. 实现 `Conversation.add_user`、`add_assistant`、`messages`。
5. 添加 conversation 顺序和副本测试。

**验证：** `uv run python -m unittest tests.test_conversation` 通过。

## T6: 实现 LLM 协议骨架

**文件：** `src/agentcode/llm/__init__.py`
**依赖：** T3、T5
**步骤：**

1. 定义 `Message`、`StreamEvent`。
2. 定义 `Provider` Protocol。
3. 实现 `new_provider(cfg)` 分派 Anthropic/OpenAI。
4. 未知协议抛防御性错误。

**验证：** `uv run python -c "from agentcode.llm import Message, StreamEvent, Provider, new_provider"` 通过。

## T7: 实现 Anthropic Provider

**文件：** `src/agentcode/llm/anthropic_provider.py`、`tests/test_llm.py`
**依赖：** T6
**步骤：**

1. 使用 `AsyncAnthropic` 构造 client。
2. 请求包含 `system=SYSTEM_PROMPT` 和完整历史。
3. `thinking=True` 时传 `thinking={"type": "enabled", "budget_tokens": 2048}`。
4. 解析正文 text delta 为 `StreamEvent(text=...)`。
5. thinking delta 丢弃。
6. 正常结束 yield `StreamEvent(done=True)`。
7. 异常 yield `StreamEvent(err=exc)`。
8. 使用 fake client 测试请求参数和 thinking 丢弃逻辑。

**验证：** `uv run python -m unittest tests.test_llm` 通过。

## T8: 实现 OpenAI Provider

**文件：** `src/agentcode/llm/openai_provider.py`、`tests/test_llm.py`
**依赖：** T6
**步骤：**

1. 使用 `AsyncOpenAI` 构造 client。
2. messages 首条插入 system prompt。
3. 使用流式 chat completions。
4. 解析正文 delta 为 `StreamEvent(text=...)`。
5. 正常结束 yield `StreamEvent(done=True)`。
6. 异常 yield `StreamEvent(err=exc)`。
7. 使用 fake client 测试请求参数。

**验证：** `uv run python -m unittest tests.test_llm` 通过。

## T9: 实现 Textual App 骨架

**文件：** `src/agentcode/tui/app.py`、`src/agentcode/tui/view.py`
**依赖：** T3、T5、T6
**步骤：**

1. 定义 `SessionState`。
2. 定义 `AgentCodeApp`。
3. compose 出 `RichLog`、动态流式区、`TextArea`、状态栏。
4. `on_mount` 写入 banner。
5. 单 provider 直进 IDLE，多 provider 进入 SELECTING。
6. Ctrl+C 调用退出。

**验证：** 使用 Textual pilot 或 smoke test 启动 App，不报错。

## T10: 实现 provider 选择界面

**文件：** `src/agentcode/tui/app.py`
**依赖：** T9
**步骤：**

1. 多 provider 时显示 `OptionList`。
2. 列出 `name (model)`。
3. Enter 选定后创建 provider。
4. 更新状态栏，隐藏选择界面，进入 IDLE。

**验证：** TUI 测试或手动启动多 provider 配置，方向键 + Enter 可进入对话。

## T11: 实现输入、流式、计时和 markdown 定型

**文件：** `src/agentcode/tui/app.py`、`src/agentcode/tui/view.py`
**依赖：** T9、T10
**步骤：**

1. Enter 提交，Alt+Enter 插入换行。
2. `/exit` 退出。
3. 提交后清空输入框，进入 STREAMING。
4. 记录 `turn_start = time.monotonic()`。
5. `set_interval` 刷新 `Imagining... (Ns)`。
6. 异步消费 `provider.stream(...)`。
7. text 增量更新动态区纯文本。
8. done 后用 Rich Markdown 渲染完整回复并写入 `RichLog`。
9. err 时写入红色错误块并回 IDLE。

**验证：** fake provider TUI 测试覆盖流式增量、done、err、计时状态。

## T12: 改造 CLI 入口

**文件：** `src/agentcode/cli.py`
**依赖：** T3、T9
**步骤：**

1. `main()` 调用 `config.load(".agentcode/config.yaml")`。
2. 配置错误打印中文可读错误并非零退出。
3. 合法配置启动 `AgentCodeApp(config.providers).run()`。

**验证：** `uv run agentcode --help`、缺配置启动错误、合法 fake 配置能进入 TUI。

## T13: 删除旧 REPL 实现并迁移测试

**文件：** `agent_code/`、旧 `tests/`
**依赖：** T12
**步骤：**

1. 删除 `agent_code/` 旧包。
2. 删除旧 REPL/CLI/provider 测试中不再适用的部分。
3. 保留并迁移有价值的配置和 provider fake 测试。
4. 确保 import 全部从 `agentcode` 包读取。

**验证：** 运行命名残留检查，确认旧包名和旧产品名不出现在源码、配置与用户文档中。

## T14: README 和文档更新

**文件：** `README.md`、`spec.md`、`plan.md`、`task.md`、`checklist.md`
**依赖：** T12
**步骤：**

1. README 写明 `agentcode` 命令。
2. README 写明 `.agentcode/config.yaml` providers 格式。
3. README 写明当前不支持工具调用、MCP、会话持久化。
4. 文档保持与实现一致。

**验证：** `sed -n '1,200p' README.md` 可看到命令、配置路径和 providers 示例。

## T15: 全量验证

**文件：** 全项目
**依赖：** T1-T14
**步骤：**

1. 运行全部单元测试。
2. 运行编译检查。
3. 运行 CLI help。
4. 使用 fake/真实 provider 做手动 TUI smoke。

**验证：**

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall src tests
uv run agentcode --help
```

## 执行顺序

```text
T1 → T2
T2 → T3 → T4
T2 → T5 → T6 → T7/T8
T3 + T5 + T6 → T9 → T10 → T11 → T12
T12 → T13 → T14 → T15
```
