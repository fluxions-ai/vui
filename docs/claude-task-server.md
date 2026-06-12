# Claude task server (optional)

A sidecar process that handles delegated, agentic work — slow lookups and tool-using tasks the main voice loop shouldn't block on. The streaming server posts to it on `:8642`; while it grinds, a parallel "thoughts" LLM call in `voice_turn.py` keeps the conversation alive with filler ("yeah, let me check…") and decides *whether* to delegate. Final text comes back via callback and gets spoken. The "task server" pill in the UI lights up when it's reachable.

## What it can do

Whatever you've connected to Claude Code on this machine. On startup, `claude_server.py` calls `discover_mcp_tools()` against the host's `~/.claude` config and registers everything it finds — no Vui-side wiring. Out of the box you also get the SDK's built-in tools: `WebSearch`, `WebFetch`, `Bash`, `Read`, `Glob`, `Grep`.

Examples of integrations people typically have hooked up via Claude Code MCPs, and the kind of voice request each unlocks:

| MCP / tool | "Hey Vui, …" |
|---|---|
| Gmail | "any unread email from Alice this week?" / "draft a reply saying I'll be 10 minutes late" |
| Google Calendar | "what's on my calendar tomorrow?" / "find a 30-min slot with Bob next week" |
| Google Drive | "pull up the Q3 planning doc and summarise the risks section" |
| Slack | "post 'standup in 5' to #eng" / "what did Carol say in the launch thread?" |
| Linear / Jira / GitHub (community MCPs) | "what tickets are assigned to me?" / "open issues on the vui repo this week" |
| Built-in `WebSearch` / `WebFetch` | "what time does the Arsenal match kick off tonight?" / "summarise this blog post: <url>" |
| Built-in `Bash` / `Read` / `Glob` | "did the nightly training run finish?" / "what's in `~/notes/IN.md`?" |

To add a new integration, hook it up in Claude Code as you normally would (`claude mcp add …` or via the desktop app) and restart `claude_server` / the `claude-task` container. To remove the agent's access to one, unhook it in Claude Code — Vui follows.

## How tool calling works

The task server (`src/vui/serving/claude_server.py`) wraps the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) as a long-lived agent loop. When the voice assistant decides a turn needs live data, the main server posts the user's request to `http://localhost:8642/task`. Inside the sidecar:

1. The model emits a `tool_use` block (e.g. `mcp__claude_ai_Gmail__search_threads`).
2. The harness executes the tool — MCP servers for Gmail / Calendar / Drive / Slack, plus built-in `WebSearch`, `WebFetch`, `Bash`, `Read`, `Glob`, `Grep`.
3. The result is fed back as a `tool_result` and the loop repeats until the model emits final text.
4. That text is POSTed back to the voice server via callback, where the TTS layer speaks it.

## Auth

One of:
1. **Claude Code subscription** (preferred) — run `claude` once on the host and log in. Credentials at `~/.claude/.credentials.json` get picked up automatically (mounted read-only into the container on the Docker path).
2. **API key** — export in your shell before bringing the service up:
   ```sh
   export ANTHROPIC_API_KEY=sk-ant-...
   # or
   export CLAUDE_CODE_OAUTH_TOKEN=...
   ```

If neither is found, the server still starts but task requests fail at runtime.

## Bring it up

**Docker:**
```sh
docker compose --profile claude up -d claude-task
docker compose logs -f claude-task
```

**Native:** install the Claude Code CLI (the Agent SDK shells out to it), then sync and run:
```sh
curl -fsSL https://claude.ai/install.sh | bash       # Linux + macOS
# or: brew install --cask claude-code                # macOS, manual upgrades

uv sync --extra claude
python -m vui.serving.claude_server                  # listens on :8642
# or, override the port:
VUI_TASK_PORT=9000 python -m vui.serving.claude_server
```

The process logs `Task server on http://0.0.0.0:8642 (model=claude-haiku-4-5-20251001)` on startup. Sanity-check it:

```sh
curl -sS -X POST http://localhost:8642/task \
  -H 'content-type: application/json' \
  -d '{"prompt": "what time is it in London right now?"}'
```

`vui-stream` defaults to `VUI_TASK_SERVER_URL=http://localhost:8642`, so with both running locally there's nothing else to wire up — point it elsewhere if you used `VUI_TASK_PORT` or moved the server to another host. Full env-var reference in [`configuration.md`](configuration.md).

## Choosing a model

The default is **Claude Haiku 4.5** (`MODEL = "claude-haiku-4-5-20251001"` in `claude_server.py`) — fast enough to keep voice-turn latency reasonable, smart enough to chain a handful of tool calls without going off the rails, and on a Pro/Max subscription it's effectively free at personal-assistant volumes. Edit the `MODEL` constant to swap in Sonnet / Opus if you need stronger reasoning.

We've found the **claude-code harness itself is a great task-management runtime even with non-Claude models** — its tool registry, MCP integration, and permissioning are some of the cleanest around. For the standard "find the latest email from X and summarise it" delegations though, Haiku on a subscription is the sweet spot.

## Non-Anthropic backends

The Agent SDK speaks Anthropic's `/v1/messages` schema, so any backend that exposes an Anthropic-compatible endpoint can stand in for the official API. Set two env vars on the `claude-task` container in `docker-compose.yml` (or in the shell that runs `python -m vui.serving.claude_server` natively):

```yaml
claude-task:
  environment:
    ANTHROPIC_BASE_URL: "https://api.z.ai/api/anthropic"   # or DeepSeek / Kimi / your gateway
    ANTHROPIC_AUTH_TOKEN: "${Z_AI_API_KEY}"
    # Optional — map Claude model names to whatever the backend serves:
    ANTHROPIC_DEFAULT_HAIKU_MODEL:  "glm-4.5-air"
    ANTHROPIC_DEFAULT_SONNET_MODEL: "glm-4.6"
    ANTHROPIC_DEFAULT_OPUS_MODEL:   "glm-4.6"
```

Common backends:

| Backend | `ANTHROPIC_BASE_URL` | Notes |
|---|---|---|
| [Ollama](https://docs.ollama.com/integrations/claude-code) ≥ 0.14 | `http://ollama:11434` | Native `/v1/messages`, fully local — see worked example below |
| z.ai (GLM 4.6) | `https://api.z.ai/api/anthropic` | Native Anthropic-compatible endpoint |
| DeepSeek | `https://api.deepseek.com/anthropic` | Set `ANTHROPIC_DEFAULT_*_MODEL` to `deepseek-chat` / `deepseek-reasoner` |
| Moonshot Kimi | `https://api.moonshot.ai/anthropic` | Native |
| [vLLM](https://docs.vllm.ai/en/stable/serving/integrations/claude_code/) | `http://your-vllm:8000` | Self-host any tool-calling-capable open-weights model |
| [LM Studio](https://lmstudio.ai/blog/claudecode) ≥ 0.4.1 | `http://host:1234` | Local inference, native `/v1/messages` |
| [LiteLLM](https://docs.litellm.ai/docs/tutorials/claude_non_anthropic_models) / [claude-code-router](https://github.com/musistudio/claude-code-router) | `http://router:port` | Per-task-type routing across many providers |

### Worked example: fully-local task server backed by Ollama

[Ollama 0.14+](https://ollama.com/blog/claude) speaks the Anthropic Messages API natively, so the existing `ollama` sidecar in `docker-compose.yml` can double as the task-server backend — no API key, no extra container. Pull a tool-calling-capable model first:

```sh
docker compose exec ollama ollama pull qwen3-coder:30b   # or gpt-oss:20b, glm-4.7-flash
```

Then point `claude-task` at it:

```yaml
claude-task:
  environment:
    ANTHROPIC_BASE_URL: "http://ollama:11434"
    ANTHROPIC_AUTH_TOKEN: "ollama"
    ANTHROPIC_API_KEY: ""
    ANTHROPIC_DEFAULT_HAIKU_MODEL:  "qwen3-coder:30b"
    ANTHROPIC_DEFAULT_SONNET_MODEL: "qwen3-coder:30b"
    ANTHROPIC_DEFAULT_OPUS_MODEL:   "qwen3-coder:30b"
    CLAUDE_CODE_ATTRIBUTION_HEADER: "0"   # critical: keeps the KV cache hot
  depends_on:
    - ollama
```

Tips:
- **Pick a model with reliable tool calling.** Without it the agent loop degrades to a chat box. `qwen3-coder`, `gpt-oss`, `glm-4.x`, and Ollama's `*:cloud` variants all support it; smaller dense models often don't.
- **Context length matters** — 32K is the floor for agent loops, 64K is the sweet spot. Set `OLLAMA_CONTEXT_LENGTH=65536` on the `ollama` service.
- **`CLAUDE_CODE_ATTRIBUTION_HEADER=0` is mandatory** for local backends — without it the per-request hash injected by the SDK invalidates Ollama's KV cache on every turn (~10× slowdown).
- **Known bug ([ollama#13949](https://github.com/ollama/ollama/issues/13949))**: the SDK occasionally calls `/v1/messages/count_tokens?beta=true`, which Ollama doesn't implement and can hang on. If the task server starts timing out, restart the `ollama` container.

Caveats:
- The backend must support **streaming tool calls** with proper argument streaming. OpenRouter is currently broken on this; route to direct providers or use a router that fixes it.
- If your gateway implements its own prompt cache, also set `CLAUDE_CODE_ATTRIBUTION_HEADER=0` so the per-request hash injected by the SDK doesn't break prefix caching.
- The `MODEL` constant in `claude_server.py` still names the Anthropic-style model the SDK requests; the `ANTHROPIC_DEFAULT_*_MODEL` env vars rewrite it on the way out.
