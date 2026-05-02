# Memory Router

> **We don't replace LLMs — we optimize how they are used.**

Memory Router is a **local-first context optimization layer** and **LLM aggregator**. It is *not* a coding assistant, an IDE, a Claude Code replacement, or a shell-runner. It does **not** execute commands, modify your files, or run git — it only manages memory and prompts.

What it does:

1. **Stores structured memory locally** in a Memory Palace (task, domain, concepts, importance).
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

## Two ways to use it

### 1. `build-context` mode — no LLM call

Memory Router builds the optimized prompt and prints it. Copy it into whatever
tool you already use (ChatGPT, Claude.ai, Claude Code, Cursor, VS Code Copilot
chat, your own scripts):

```bash
memory-router build-context "Explain bond convexity again"
```

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

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  memory-router CLI                      │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
   ┌──────────┐    ┌────────────────┐    ┌──────────────┐
   │classifier│ →  │ context_builder│ →  │token_optimize│
   └──────────┘    └────────┬───────┘    └──────┬───────┘
        ▲                   │                   │
        │                   ▼                   ▼
   ┌──────────┐       ┌──────────┐         ┌────────┐
   │  query   │       │  Memory  │         │ Router │
   └──────────┘       │  Palace  │         └───┬────┘
                      │ (sqlite) │             │
                      └──────────┘             ▼
                                       ┌────────────────┐
                                       │  Providers     │
                                       │  ─ Ollama      │
                                       │  ─ OpenAI      │
                                       │  ─ Anthropic   │
                                       │  ─ Ruflo       │
                                       └────────────────┘
```

## How token saving works

Naive chat clients send **every prior message** with each new query. After 30 turns that's thousands of redundant tokens.

Memory Router builds the context as:

- 0–N system notes from the **Memory Palace** (top-K relevant memories)
- 1 short summary of older chat turns
- the **last 5–8 messages** verbatim
- the current query

And then a token-budget pass trims anything over your limit. The savings shown in the output compares this trimmed context against what a "send-everything" approach would have used.

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
