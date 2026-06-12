# AgentCode

AgentCode 是一个 Claude Code 风格的终端 AI 客户端。当前阶段支持纯文本多轮对话、OpenAI/Anthropic 流式输出、Claude extended thinking，以及一轮工具调用回灌。

当前内置工具为 `read`、`write`、`edit`、`bash`、`find`、`grep`。自动多轮 Agent Loop、MCP、权限系统和会话持久化暂未实现。

## 启动

```bash
uv run agentcode
```

也可以使用模块入口：

```bash
uv run python -m agentcode
```

## 配置

运行配置位于：

```text
.agentcode/config.yaml
```

该文件包含真实密钥，已被 `.gitignore` 忽略。先复制模板：

```bash
cp .agentcode/config.yaml.example .agentcode/config.yaml
```

配置格式：

```yaml
providers:
  - name: "Anthropic Claude"
    protocol: anthropic
    model: claude-sonnet-4-5-20250929
    base_url: https://api.anthropic.com
    api_key: replace-with-your-anthropic-api-key
    thinking: true

  - name: "OpenAI"
    protocol: openai
    model: gpt-4o
    base_url: https://api.openai.com/v1
    api_key: replace-with-your-openai-api-key
    thinking: false
```

如果只配置一个 provider，AgentCode 会直接进入对话；如果配置多个 provider，启动后会先出现选择界面。
