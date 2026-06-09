# 多协议 LLM 终端对话客户端 Checklist

> 每一项通过运行代码或观察行为来验证，聚焦系统行为。

## 实现完整性

- [ ] 合法 `.agentcode/config.yaml` 能解析出 providers 列表（验证：配置单测 + 启动进入 TUI）。
- [ ] 缺密钥、非法 protocol、文件缺失时给出可读错误并非零退出，无未捕获堆栈（验证：分别运行坏配置）。
- [ ] 单 provider 配置时启动直接进入对话（验证：单条配置运行）。
- [ ] 多 provider 配置时出现方向键选择列表，选定后进入对话（验证：两条配置运行，上下选择 + Enter）。
- [ ] 底部状态栏显示活动 provider 名称和模型（验证：启动后观察状态栏）。
- [ ] 发出的请求包含内置 system prompt 与完整历史（验证：fake provider 断言请求参数）。
- [ ] Anthropic `thinking: true` 时启用扩展思考，thinking 增量不显示（验证：fake event 覆盖 thinking delta）。
- [ ] 回复以纯文本逐字流式出现（验证：fake provider 或真实长回复观察）。
- [ ] 回复结束后整段以 markdown 渲染，代码块、列表、强调正确（验证：让 fake/真实 provider 返回 markdown）。
- [ ] Alt+Enter 可换行，Enter 提交，提交后输入框清空（验证：TUI 手动或 pilot 测试）。
- [ ] 自提交即显示 `Imagining... (Ns)`，秒数递增，结束后显示总耗时（验证：慢流式 fake provider）。
- [ ] 错误 key 或不存在模型时，错误在对话区以红色/可区分样式显示，程序不退出（验证：fake provider err 或真实坏 key）。
- [ ] `/exit` 与 Ctrl+C 均能安全退出，终端恢复正常（验证：手动退出两种方式）。
- [ ] 启动界面包含猫 banner、应用名版本、cwd、就绪提示行、带 `❯` 与占位符的输入框、状态栏（验证：启动截图/观察）。

## 集成

- [ ] TUI 通过统一 Provider Protocol 驱动 Anthropic/OpenAI，切换协议不改变上层交互（验证：fake provider 和真实配置各跑一轮）。
- [ ] 多轮上下文携带：先告知信息、后追问，模型能引用前文；退出再启动后历史为空（验证：真实或 fake provider）。
- [ ] 等待与流式期间界面保持响应，可滚动、不冻结（验证：长回复期间滚动对话区）。
- [ ] `base_url` 覆盖生效（验证：fake SDK 构造参数或兼容端点真实运行）。
- [ ] Markdown 和布局在窄终端下自适应（验证：调整终端宽度观察）。
- [ ] API key 不在 UI、错误输出或日志中明文显示（验证：使用明显测试 key 后检索输出）。

## 编译与测试

- [ ] `uv run python -m unittest discover -s tests` 通过。
- [ ] `uv run python -m compileall src tests` 通过。
- [ ] `uv run agentcode --help` 可用。
- [ ] `uv run python -m agentcode --help` 可用。
- [ ] 源码、配置与用户文档无旧包名或旧产品名残留（验证：运行命名残留检查）。

## 端到端场景

- [ ] 场景 1：Anthropic 单 provider 多轮对话：启动直接进入 TUI，连续两轮，流式 + 计时 + markdown 定型，`/exit` 退出。
- [ ] 场景 2：OpenAI 单 provider 流式 markdown：发一条含代码块请求，流式逐字后 markdown 渲染正确。
- [ ] 场景 3：多 provider 选择：两条配置启动，出现列表，选第二条，状态栏显示其 name/model，正常对话。
- [ ] 场景 4：错误恢复：错误 key 触发失败，对话区显示错误，程序不退出，可继续下一轮。
- [ ] 场景 5：无真实 API key 时的可验证范围：配置、conversation、provider fake 测试、TUI fake 测试、编译和 CLI help 通过；真实 E2E 明确标注未执行。
