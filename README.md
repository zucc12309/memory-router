# Memory Router

> **We don't replace LLMs — we optimize how they are used.**

Memory Router is a **local-first context optimization layer** and **LLM aggregator**. It is *not* a coding assistant, an IDE, a Claude Code replacement, or a shell-runner. It does **not** execute commands, modify your files, or run git — it only manages memory and prompts.

What it does:

1. **Stores structured memory locally** in a Memory Palace (task, domain, concepts, importance).
   It can also automatically promote useful completed turns into that Memory Palace.
2. **Retrieves only the relevant context** for each query instead of dragging your full chat history.
3. **Builds an optimized prompt** with the right memories + a short summary + the last few turns.
4. **Optionally routes the prompt** to the best available LLM — local (Ollama), OpenAI, Anthropic, Google Gemini, or an optional Ruflo backend.
5. Or just **prints the optimized prompt** for you to paste into ChatGPT, Claude.ai, Claude Code, VS Code, or any tool you already use.

Memory Router works *alongside* your existing tools. Your memory stays on your machine; only the trimmed context for the current question is sent to whichever model you pick.

---

## Why local-first?

Most "AI memory" tools synchronize your conversations to a remote server so they can do retrieval. That's a privacy footgun: your past questions, code, finance notes, and half-formed ideas now live somewhere you don't control.

Memory Router keeps the entire memory layer on your laptop:

- Conversations: `~/.memory-router/conversations.sqlite`
- Memories: `~/.memory-router/memories.sqlite`
- API keys: OS keychain (macOS Keychain, Windows Credential Locker, Linux Secret Service), with a 0600-permission fallback file
- Vector index (when you enable one): `~/.memory-router/vector_index/`

You can wipe everything with `rm -rf ~/.memory-router`. There is no cloud component to opt out of.

## API keys vs ChatGPT/Claude subscriptions

These are different products and Memory Router uses the API path:

| | Subscription (ChatGPT Plus, Claude Pro) | API key |
|---|---|---|
| What it is | A web/app product you log in to | A developer credential billed per token |
| Where it works | chat.openai.com, claude.ai | Any client that calls the provider's API |
| Memory Router | ❌ Cannot use a subscription | ✅ Uses your API key |
| Billing | Flat monthly fee | Pay-per-use (tokens in + tokens out) |

You'll need API keys from [platform.openai.com](https://platform.openai.com), [console.anthropic.com](https://console.anthropic.com), or [aistudio.google.com](https://aistudio.google.com/app/apikey) (Gemini) to use those providers. For local-only mode you don't need any keys — just [Ollama](https://ollama.com).

---

## Install

```bash
pip install memory-router

# Optional providers
pip install "memory-router[openai]"
pip install "memory-router[anthropic]"
pip install "memory-router[gemini]"
pip install "memory-router[all]"
```

From source:

```bash
git clone https://github.com/yourusername/memory-router
cd memory-router
pip install -e .
```

### Testing without API keys

You can test the project completely offline:

```bash
./.venv/bin/pytest -q
memory-router init   # choose local mode
memory-router build-context "Explain this code"
memory-router memory add "Prefer pytest for tests" --domain software --task code
memory-router benchmark --no-run
```

The unit tests do not require OpenAI, Anthropic, or Gemini keys. Cloud keys are
only needed if you choose `api` or cloud-backed `hybrid` routing.

If you have Ollama installed, you can also run the benchmark against a local
model without paying for API usage:

```bash
memory-router benchmark --local
```

The benchmark prints raw prompt tokens, optimized prompt tokens, estimated
savings, and a simple quality score when a model backend is available.

## Setup

```bash
memory-router init
```

You'll be asked to pick a mode:

- **local** — only local models (Ollama). No API keys needed. Most private.
- **api** — OpenAI / Anthropic / Gemini only. Requires API keys.
- **hybrid** — local for simple queries, API for complex ones. Recommended.
- **ruflo** — adds Ruflo as a provider for multi-agent / agentic workflows.

Then add credentials separately so they go straight to your OS keychain:

```bash
memory-router auth openai
memory-router auth anthropic
memory-router auth gemini
```

---

## Three ways to use it

### 1. `build-context` mode — no LLM call

Memory Router builds the optimized prompt and prints it. Copy it into whatever
tool you already use (ChatGPT, Claude.ai, Claude Code, Cursor, VS Code Copilot
chat, your own scripts):

```bash
memory-router build-context "Explain bond convexity again"
```

When you use `memory-router ask ...`, useful turns can be auto-saved into the
Memory Palace so future questions can reuse them without manual bookkeeping.

Example output:

```
Relevant memory used:
- [finance/explain] User previously studied duration and convexity
- [prefs/general]   User prefers simple explanations with examples

task=explain  domain=finance  concepts=['bond','convexity']  tokens_sent≈410  saved≈86%

Optimized prompt:
Relevant memories from past conversations:
- [finance/explain] User previously studied duration and convexity
- [prefs/general]   User prefers simple explanations with examples

---

Explain bond convexity again
```

No API keys are needed for this mode. No network calls happen. Use
`--show-messages` to print a role-tagged message list instead of a flat prompt.

### 2. Provider mode — route and answer

When you do want Memory Router to call a model for you:

```bash
memory-router "Explain bond convexity"
memory-router --no-memory "Private question — don't use my memory palace"
memory-router --local "Stay on local model"

# Memory palace
memory-router memory palace
memory-router memory list
memory-router memory add "User prefers concise answers" --domain prefs --importance 0.9
memory-router memory delete 3
memory-router memory clear

# Config
memory-router config show
memory-router config set mode hybrid
memory-router config set token_budget 6000
```

### Example output

```
Using: claude-sonnet-4-6
Memory used: Explain > Finance > fixed-income, duration
Estimated tokens saved: 84%

╭─ Answer ───────────────────────────────────────────╮
│ Convexity measures the curvature of a bond's       │
│ price-yield relationship...                        │
╰────────────────────────────────────────────────────╯
```

### 3. MCP server — plug into Claude Code, Cursor, Cline, Continue

Memory Router can run as a [Model Context Protocol](https://modelcontextprotocol.io) server. Any MCP-compatible client can call its tools to retrieve memories, store new ones, build optimized contexts, and capture useful turns — without you copy-pasting anything.

**One-time install** — install globally so MCP clients can find it on PATH (clients spawn the server *outside* any venv you may have active):

```bash
# pipx is the standard way to install Python CLIs globally
brew install pipx                                     # macOS
# or: python3 -m pip install --user pipx              # any platform
pipx ensurepath                                       # adds ~/.local/bin to PATH

# Install Memory Router with the MCP + provider extras
pipx install memory-router
pipx inject memory-router "memory-router[mcp,gemini,openai,anthropic]"

memory-router init
which memory-router      # confirm it's on PATH (e.g. ~/.local/bin/memory-router)
```

> **Why pipx and not `pip install -e`?** MCP clients (Claude Code, Cursor, etc.) spawn the server as a fresh subprocess without your venv active. If `memory-router` only lives inside a venv, the client can't find it. `pipx` installs it system-wide while keeping its dependencies isolated.

**Register with your client** (pick one or more):

```bash
# Claude Code — use --scope user so it's available in every project
claude mcp add --scope user memory-router -- memory-router mcp serve

# Cursor — add to ~/.cursor/mcp.json
# { "mcpServers": { "memory-router": { "command": "memory-router", "args": ["mcp", "serve"] } } }

# Cline (VS Code) — Settings → MCP Servers, add:
# Name: memory-router | Command: memory-router | Args: mcp serve

# Continue — add to ~/.continue/config.yaml
# mcpServers:
#   - name: memory-router
#     command: memory-router
#     args: ["mcp", "serve"]
```

> **Restart your client** after registering. MCP servers are loaded at session start; an already-open Claude Code / Cursor / etc. won't pick up the new config until you restart.

> **Working from a clone?** If you'd rather not install globally, register with the absolute path to your venv binary instead — e.g. `claude mcp add --scope user memory-router -- /Users/you/Documents/GitHub/memory-router/.venv/bin/memory-router mcp serve`. This works but breaks if you delete the venv.

Now in any session of those tools, the agent can call:

| Tool | What it does |
|---|---|
| `memory_search(query, top_k=5)` | Retrieve top-K relevant memories for a query |
| `memory_store(content, domain, importance, ...)` | Save a durable fact |
| `memory_list(limit=20)` | List memories ordered by importance + recency |
| `memory_palace()` | Show domain → task hierarchy |
| `memory_delete(memory_id)` | Delete by id |
| `memory_capture(query, answer, ...)` | Promote a useful turn to long-term memory |
| `build_context(query)` | Build the optimized prompt + report saved tokens |
| `log_turn(query, answer)` | Record a Q&A into the conversation log |
| `stats_summary()` | Cumulative token-saving stats |
| `stats_reset()` | Wipe stats |

Every `build_context` call records its token impact into `stats.sqlite`, so you can run `memory-router stats` from your terminal at any time and see how much the layer is saving across all your sessions.

#### Verify it's working

After restarting your client, ask in chat:

```
List the memory-router MCP tools that are available.
```

The agent should ToolSearch and find the 10 tools above. Then a real test:

```
Remember that I always use python-jose for JWT, never pyjwt.
```

It should call `memory_store`. Confirm from your terminal:

```bash
memory-router memory list   # should show the new memory
memory-router stats         # should start showing call counts
```

#### Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: memory-router` when client starts | Memory Router isn't on system PATH. Use `pipx install memory-router` (above) or register with the absolute path to your venv binary. |
| Tools don't appear in the client | Restart the client. MCP servers load at session start. |
| Tools available in one project but not another | You used the default `local` scope. Re-register with `claude mcp add --scope user ...`. |
| `mcp` package missing error on launch | Install the extra: `pipx inject memory-router "memory-router[mcp]"`. |
| Need to start over | `claude mcp remove memory-router` and re-add. |

#### Optional: auto-augment every Claude Code prompt

If you want every prompt you type in Claude Code to silently get memory context (without the agent having to call `memory_search` first), add this to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "command": "memory-router build-context --stdin --json"
      }
    ]
  }
}
```

The hook is opt-in. Without it, the agent decides when to call `memory_search` based on your prompt; with it, every prompt gets pre-augmented automatically.

---


## How token saving works

Naive chat clients send **every prior message** with each new query. After 30 turns that's thousands of redundant tokens.

Memory Router builds the context as:

- 0–N system notes from the **Memory Palace** (top-K relevant memories)
- 1 short summary of older chat turns
- the **last 5–8 messages** verbatim
- the current query

And then a token-budget pass trims anything over your limit. The savings shown in the output compares this trimmed context against what a "send-everything" approach would have used.

## Tracking your savings

Every CLI provider call and every MCP `build_context` invocation appends a row into `~/.memory-router/stats.sqlite` — token counts, memories used, provider, model, and estimated cost. No prompt content, no answer text, just the numbers.

See it any time:

```bash
memory-router stats
```

Sample output:

```
╭─ Memory Router — Cumulative Savings ─────────────────────╮
│              Calls tracked:  47                          │
│ Tokens that would have been sent:  324,810               │
│              Tokens actually sent:   38,420              │
│                       Tokens saved:  286,390  (88%)      │
│             Output tokens received:   67,210             │
│                  Memories injected:  131                 │
│              Estimated cost (real):  $0.124              │
╰──────────────────────────────────────────────────────────╯

By provider
  gemini       42 calls   …
  openai        4 calls   …
  ollama        1 calls   …

By kind
  cli_ask              35 calls   …
  mcp_build_context    12 calls   …
```

`memory-router stats --reset` wipes the table. `--json` prints raw JSON for piping into other tools.

## Proof: real measured savings

Numbers from a real session, verified against Google's tokenizer (not estimated). The setup was a 4-turn conversation with `gemini-2.5-flash`: build a FastAPI URL shortener, then follow up with three more queries. Memory Palace was seeded with two memories about the user's stack and code preferences.

### Per-query, by the third follow-up

The third follow-up — *"Write a Dockerfile for it"* — sent against the same conversation history:

| Metric | Naive (full history) | Memory Router | Saving |
|---|---|---|---|
| **Input tokens (real)** | **10,142** | **1,224** | **88%** |
| Output tokens | 1,570 | 1,733 | — |
| **Total cost** | **$0.0016** | **$0.0008** | **~50%** |

Across all four turns of the session: **~17,000 naive input tokens vs ~1,800 sent — about 89% less context shipped.**

### And the answers got *better*

When called naively with the full 10k-token history, the model replied:

> "You're asking for a Dockerfile again! I've provided one twice already. Is there anything specific you'd like to change..."

It got confused by the bloated history. The same model, given Memory Router's 1.2k-token optimized prompt, returned a clean multi-stage Dockerfile right away. This is the well-known "lost in the middle" effect: long contexts dilute attention. Memory Router avoids it by design.

So the real story is two-fold:

1. **~88% fewer input tokens** → ~50% cheaper per query and orders of magnitude cheaper at scale.
2. **More focused answers** because the model isn't wading through stale turns to find the relevant context.

### Reproduce it yourself

The repo includes two scripts under `scripts/` that you can run against your own `~/.memory-router/conversations.sqlite`:

```bash
# 1. Run a query through Memory Router (note the "Input tokens (real)" line)
memory-router --model gemini-2.5-flash "Write a Dockerfile for it"

# 2. Run the same query naively, with the full history, no Memory Router
python scripts/naive_call.py "Write a Dockerfile for it"
```

The second script reads the same conversation history and calls Gemini directly. Compare the two `Input tokens (real)` numbers — that's your verified saving for that query.

---

## Project layout

```
memory_router/
├── cli.py                  # Typer CLI entry point
├── config.py               # ~/.memory-router/config.yaml
├── classifier.py           # rule-based task/domain/concept extraction
├── context_builder.py      # assembles trimmed messages for the LLM
├── token_optimizer.py      # enforces token budget
├── router.py               # picks (provider, model) based on rules
├── memory/
│   ├── palace.py           # domain → task hierarchy view
│   ├── sqlite_store.py     # MemoryStore + ConversationStore
│   ├── vector_store.py     # extension point for FAISS/Chroma/etc.
│   └── summarizer.py       # cheap deterministic summarizer
├── providers/
│   ├── base.py             # BaseProvider interface
│   ├── ollama_provider.py  # local Ollama via HTTP
│   ├── openai_provider.py  # lazy OpenAI SDK
│   ├── anthropic_provider.py
│   └── ruflo_provider.py   # optional multi-agent
├── security/
│   └── keychain.py         # OS keychain + 0600-fallback
└── utils/
    └── tokens.py           # estimator + savings helper
```

---

## What Memory Router is *not*

To keep the scope honest:

- ❌ Not a coding assistant or pair-programmer
- ❌ Does not execute shell commands
- ❌ Does not modify your files
- ❌ Does not run git
- ❌ Not an IDE or Claude Code replacement

It is a **memory + prompt-optimization layer**. Use it alongside your existing
tools — that's the point.

## Working with code answers

Memory Router prints answers to your terminal. **It does not create files, run shell commands, install dependencies, modify your repo, or run git.** If the model returns ten code blocks, you get ten code blocks of text — copy them into your editor yourself, or pipe them to a tool whose job is execution.

This is intentional, not a missing feature. Two reasons:

1. **Trust boundary.** Memory Router is meant to be a thin, auditable layer: context in, text out. The moment a tool can write files based on LLM output, you've handed the model write-access to your filesystem. We don't.
2. **Better tools already exist** for code execution — Claude Code, Cursor, Aider, Copilot Workspace. They handle diff review, sandboxing, and undo properly. Memory Router would be a worse version of those if it tried.

The mental model: **Memory Router is the prompt-prep layer. Your IDE / Claude Code / hands are the execution layer.**

### Three workflows for code answers

**1. Save the answer and copy code blocks manually** — fine for one-off scaffolding:

```bash
memory-router --model gemini-2.5-flash "Build a FastAPI URL shortener" > plan.md
# Open plan.md in your editor, copy code blocks into real files
```

**2. Build the optimized prompt and hand it to a tool that executes** — the recommended workflow:

```bash
memory-router build-context "Add JWT auth to my URL shortener" | pbcopy
# Paste into Claude Code (or Cursor's chat) — it can write files, run tests, etc.
# You get Memory Router's context optimization PLUS your IDE's execution.
```

**3. Pair with `aider` or any CLI agent** — they're designed for repo edits; Memory Router can prepare the prompt:

```bash
memory-router build-context "Refactor the auth flow to use refresh tokens" > /tmp/prompt.txt
aider --message-file /tmp/prompt.txt
```

## Roadmap

- [ ] Real vector embeddings (FAISS / sqlite-vec / Chroma)
- [ ] LLM-backed concept extraction as a fallback to the rule-based classifier
- [ ] LLM-based summarizer (opt-in, runs on local model)
- [ ] Per-domain importance decay
- [ ] Streaming responses
- [ ] `memory-router import` for ChatGPT/Claude exports
- [ ] First-class Ruflo plugin once the public API stabilizes
- [ ] `tiktoken` integration for precise OpenAI token counts

---

## License

MIT.
