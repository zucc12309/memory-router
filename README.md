# Memory Router

> **We don't replace LLMs — we optimize how they are used.**

Memory Router is a **local-first context optimization layer** and **LLM aggregator**. It is *not* a coding assistant, an IDE, a Claude Code replacement, or a shell-runner. It does **not** execute commands, modify your files, or run git — it only manages memory and prompts.

What it does:

1. **Stores structured memory locally** in a Memory Palace (task, domain, concepts, importance) with FTS5 full-text search, confidence decay, and mycelium-inspired associative retrieval.
2. **Retrieves only the relevant context** for each query using priority-scored assembly instead of dragging your full chat history.
3. **Builds an optimized prompt** with the right memories + working memory + a short summary + the last few turns — typically saving **80–90% of input tokens**.
4. **Routes the prompt** to the best available LLM — local (Ollama), OpenAI, Anthropic, Google Gemini — with automatic fallback and adaptive learning from past outcomes.
5. **Streams responses** from all providers with real-time token display.

Memory Router works *alongside* your existing tools. Your memory stays on your machine; only the trimmed context for the current question is sent to whichever model you pick.

---

## What's New in v0.2

| Feature | Description |
|---|---|
| **FTS5 Search** | Full-text search replaces O(n) table scan — instant retrieval at any scale |
| **Mycelium Network** | Bio-inspired associative memory graph with spreading activation |
| **Working Memory** | Session-scoped scratchpad with relevance decay and LRU eviction |
| **Memory Decay** | Exponential confidence decay with reinforcement on usage |
| **Priority Context** | Each message block scored 0.0–1.0; low-priority dropped first |
| **Adaptive Routing** | Learns from outcomes (quality/cost/latency) to improve model selection |
| **Streaming** | Real-time token streaming across all 4 providers |
| **Fallback Routing** | Automatic provider failover on errors |
| **Import/Export** | Import from ChatGPT, Claude, or generic JSON; encrypted export |
| **Consolidation** | Find and merge near-duplicate memories automatically |
| **Semantic Dedup** | Jaccard-based similarity search before storing |
| **Encrypted Export** | AES-256-GCM for exported/imported memory files with machine-derived keys |
| **HMAC Integrity** | Tamper detection on the secrets fallback file |
| **Structured Logging** | JSON-formatted rotating logs under `~/.memory-router/logs/` |
| **Health Checks** | Programmatic `check_health()` and `memory-router doctor` |
| **Config Validation** | Enum/range validation on all config fields |
| **tiktoken** | Accurate OpenAI token counting with graceful heuristic fallback |
| **368+ Tests** | Comprehensive test suite (72% coverage) covering all modules |

---

## Why local-first?

Most "AI memory" tools synchronize your conversations to a remote server so they can do retrieval. That's a privacy footgun: your past questions, code, finance notes, and half-formed ideas now live somewhere you don't control.

Memory Router keeps the entire memory layer on your laptop:

- Conversations: `~/.memory-router/conversations.sqlite`
- Memories: `~/.memory-router/memories.sqlite`
- API keys: OS keychain (macOS Keychain, Windows Credential Locker, Linux Secret Service), with HMAC-protected 0600-permission fallback
- Vector index: `~/.memory-router/vector_index/`
- Route history: `~/.memory-router/route_history.sqlite` (adaptive routing)
- Logs: `~/.memory-router/logs/memory-router.log` (JSON-structured, 5 MB rotation)

You can wipe everything with `rm -rf ~/.memory-router`. There is no cloud component to opt out of.

## API keys vs ChatGPT/Claude subscriptions

These are different products and Memory Router uses the API path:

| | Subscription (ChatGPT Plus, Claude Pro) | API key |
|---|---|---|
| What it is | A web/app product you log in to | A developer credential billed per token |
| Where it works | chat.openai.com, claude.ai | Any client that calls the provider's API |
| Memory Router | ❌ Cannot use a subscription | ✅ Uses your API key |
| Billing | Flat monthly fee | Pay-per-use (tokens in + tokens out) |

You'll need API keys from [platform.openai.com](https://platform.openai.com), [console.anthropic.com](https://console.anthropic.com), or [aistudio.google.com](https://aistudio.google.com/app/apikey) (Gemini) to use those providers. For local-only mode you don't need any keys — just [Ollama](https://ollama.com). Memory Router will start Ollama in the background for you on the first local request, pull the selected model if needed, and during setup suggest a local model based on your machine's RAM and CPU.

---

## Install

```bash
pip install memory-router

# Optional providers & extras
pip install "memory-router[openai]"
pip install "memory-router[anthropic]"
pip install "memory-router[gemini]"
pip install "memory-router[mcp]"          # MCP server (Claude Code, Cursor, ...)
pip install "memory-router[encryption]"   # AES-256-GCM encrypted export/import
pip install "memory-router[tiktoken]"     # accurate OpenAI token counting
pip install "memory-router[numpy]"        # faster vector similarity backend
pip install "memory-router[all]"          # everything
```

From source:

```bash
git clone https://github.com/zucc12309/memory-router.git
cd memory-router
pip install -e ".[dev]"
```

### Testing without API keys

```bash
pytest -q                                          # 368 tests, no API keys needed
memory-router init                                 # choose local mode
memory-router build-context "Explain this code"
memory-router memory add "Prefer pytest" --domain software --task code
memory-router benchmark --no-run
```

If you have Ollama installed, run benchmarks against a local model:

```bash
memory-router benchmark --local
```

## Setup

```bash
memory-router init
```

You'll be asked to pick a mode:

- **local** — only local models (Ollama). No API keys needed. Most private. Ollama auto-starts in the background on first use, pulls the selected model if needed, and setup suggests a model from your system specs.
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

## Usage

### 1. `build-context` mode — no LLM call

Build the optimized prompt and print it. Copy into whatever tool you use:

```bash
memory-router build-context "Explain bond convexity again"
```

Example output:

```
Relevant memory used:
- [finance/explain] User previously studied duration and convexity
- [prefs/general]   User prefers simple explanations with examples

task=explain  domain=finance  concepts=['bond','convexity']  tokens_sent≈410  saved≈86%
```

### 2. Provider mode — route and answer

```bash
memory-router "Explain bond convexity"                    # auto-route
memory-router "Explain this" --stream                     # stream response
memory-router --no-memory "Private question"              # skip memory
memory-router --local "Stay on local model"               # force Ollama (auto-starts if needed)
memory-router --provider openai --model gpt-4o "Code it"  # pin provider
```

### 3. Memory management

```bash
# Core CRUD
memory-router memory list
memory-router memory add "Prefer pytest" --domain software --task code --importance 0.9
memory-router memory delete 3
memory-router memory palace               # hierarchical domain→task view
memory-router memory search "auth tokens" # FTS5 + keyword search

# Memory health
memory-router memory decay                # show decay stats
memory-router memory decay --prune        # remove stale memories

# Import/Export
memory-router memory export backup.json             # export all memories
memory-router memory import conversations.json      # ChatGPT/Claude/generic auto-detect

# Dedup & consolidation
memory-router memory similar "user prefers dark mode"  # find near-duplicates
memory-router memory consolidate                       # preview merges (dry run)
memory-router memory consolidate --apply               # merge near-duplicates

# Mycelium network
memory-router memory network              # show associative graph stats
```

### 4. Adaptive routing

When enabled, Memory Router learns from past outcomes to improve model selection:

```bash
memory-router config set adaptive_routing true
memory-router routing-report              # see provider performance data
```

### 5. Config

```bash
memory-router config show
memory-router config set mode hybrid
memory-router config set token_budget 8000
memory-router config set mycelium_enabled true
memory-router config set encryption_enabled true
memory-router config set local_model llama3.1:8b   # choose the default Ollama model
```

### 6. Diagnostics

```bash
memory-router doctor    # system health check
memory-router stats     # cumulative token savings
memory-router stats --json
memory-router stats --reset
```

---

## Obsidian export — see your memory as a knowledge graph

Memory Router can project the Memory Palace into an [Obsidian](https://obsidian.md)
vault: human-readable Markdown notes you can browse, edit, and explore in
**Graph View** — with your mycelium associations rendered as `[[wikilinks]]`.

**SQLite stays the source of truth.** The vault is a one-way, regenerable
projection. Retrieval never reads it; disabling the feature changes nothing.
It's **off by default**.

```bash
# One-time: scaffold the vault, save config, enable the feature
memory-router memory obsidian init --vault ~/Documents/MemoryRouterVault

# Export consolidated knowledge notes (Projects/Research/Decisions/People/…)
memory-router memory obsidian export

# Export knowledge notes + one note per raw memory
memory-router memory obsidian export --all

# Export a single project's notes
memory-router memory obsidian export --project RideCompare

# Show status: enabled, vault path, memory count, notes exported, last export
memory-router memory obsidian status
```

**Vault structure:**

```
MemoryRouterVault/
├── 00_Inbox/            # unmatched domains
├── 01_Projects/         # software/app/product memories
├── 02_Research/         # research/science/ML memories
├── 03_Decisions/        # architecture/ADR/tradeoff memories
├── 04_People/           # people/contacts
├── 05_Conversations/    # chat/thread memories
├── 06_Daily/            # reserved for daily notes
├── 90_Raw_Memories/     # one lossless note per memory (opt-in)
├── 99_Archive/          # backups before overwrite
├── Memory Router Index.md
└── README.md            # privacy warning + .gitignore
```

**Graph View:** knowledge notes link to their related concepts and source
memories; raw memory notes link to mycelium neighbours. Open Obsidian's Graph
View and the associations Memory Router learned become a navigable web. Edge
weights are preserved as comments so nothing is lost:

```markdown
## Related Concepts
- [[CatBoost]] <!-- weight 0.82 -->
- [[Flutter]]  <!-- weight 0.74 -->
```

**Privacy:** the vault may contain sensitive memories. `init` writes a `.gitignore`
that ignores the whole vault and a README warning. With
`obsidian_redact_sensitive_data` on (default), API keys, tokens, JWTs, and emails
are stripped from notes before they're written (one-way — the originals stay in
SQLite). Do **not** commit or cloud-sync the vault without reviewing it.

Configure via `memory-router config set`:

| Key | Default | Meaning |
|---|---|---|
| `obsidian_enabled` | `false` | Master switch (set by `init`) |
| `obsidian_vault_path` | `""` | Vault directory (set by `init`) |
| `obsidian_export_mode` | `knowledge_notes` | `knowledge_notes` \| `raw` \| `both` |
| `obsidian_edge_threshold` | `0.6` | Min mycelium weight to emit as a wikilink |
| `obsidian_generate_backlinks` | `true` | Render mycelium edges as `[[wikilinks]]` |
| `obsidian_include_raw_memories` | `false` | Also emit one note per memory |
| `obsidian_redact_sensitive_data` | `true` | Strip secrets/PII before writing |

---

### MCP server — plug into Claude Code, Cursor, Cline, Continue

Memory Router runs as a [Model Context Protocol](https://modelcontextprotocol.io) server. Any MCP-compatible client can call its tools.

**One-time install:**

```bash
# pipx for global CLI access
brew install pipx && pipx ensurepath

# Install with extras
pipx install memory-router
pipx inject memory-router "memory-router[mcp,gemini,openai,anthropic]"

memory-router init
```

**Register with your client:**

```bash
# Claude Code
claude mcp add --scope user memory-router -- memory-router mcp serve

# Cursor — add to ~/.cursor/mcp.json
# { "mcpServers": { "memory-router": { "command": "memory-router", "args": ["mcp", "serve"] } } }

# Cline (VS Code) — Settings → MCP Servers
# Ccommand: memory-router | Args: mcp serve
```

**Available MCP tools (20):**

| Tool | What it does |
|---|---|
| `memory_search(query, top_k)` | Retrieve relevant memories with FTS5 + mycelium spread |
| `memory_store(content, domain, importance, ...)` | Save a durable fact |
| `memory_list(limit)` | List memories by importance + recency |
| `memory_palace()` | Show domain → task hierarchy |
| `memory_delete(memory_id)` | Delete by id (+ mycelium edge cleanup) |
| `memory_capture(query, answer, ...)` | Promote a turn to long-term memory |
| `memory_find_similar(content, threshold)` | Find near-duplicate memories |
| `memory_consolidate(threshold, dry_run)` | Merge near-duplicate memories |
| `memory_decay_stats()` | Memory health (confidence distribution) |
| `memory_prune(threshold, min_age)` | Remove stale memories |
| `build_context(query, session_id)` | Build optimized prompt + report savings |
| `log_turn(query, answer, session_id)` | Record Q&A into conversation log |
| `working_memory_set(key, value, session_id)` | Store session-scoped context |
| `working_memory_get(key, session_id)` | Retrieve session context |
| `working_memory_snapshot(session_id)` | Dump current working memory |
| `mycelium_stats()` | Associative network health |
| `mycelium_neighbors(memory_id, limit)` | Direct neighbors in the graph |
| `stats_summary()` | Cumulative token-saving stats |
| `stats_reset()` | Reset cumulative usage stats |
| `health_check()` | System diagnostics |

All tools are rate-limited (configurable via `mcp_rate_limit`), use singleton connection pooling, and validate inputs.

#### Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: memory-router` | Use `pipx install memory-router` or register with absolute venv path |
| Tools don't appear | Restart the client (MCP servers load at session start) |
| Tools available in one project only | Re-register with `--scope user` |
| `mcp` package missing | `pipx inject memory-router "memory-router[mcp]"` |

---

## How token saving works

Naive chat clients send **every prior message** with each new query. After 30 turns that's thousands of redundant tokens.

Memory Router builds context with **priority-scored assembly**:

| Block | Priority | Description |
|---|---|---|
| Current query | 1.0 | Always kept — never dropped |
| System instructions | 0.95 | Coding mode guidance |
| Working memory | 0.90 | Current session context (active file, variables, etc.) |
| Memory Palace | 0.80 | Top-K relevant memories + mycelium spread |
| Recent turns | 0.70–0.20 | Last K turns, decaying by age |
| Summary | 0.30 | Compressed older history |

The token optimizer drops lowest-priority blocks first and compresses mid-priority ones to fit the budget. This replaces v1's blind positional trimming.

## Proof: real measured savings

Numbers from a real 4-turn session with `gemini-2.5-flash`:

| Metric | Naive (full history) | Memory Router | Saving |
|---|---|---|---|
| **Input tokens** | **10,142** | **1,224** | **88%** |
| Output tokens | 1,570 | 1,733 | — |
| **Total cost** | **$0.0016** | **$0.0008** | **~50%** |

The naive prompt caused the model to say *"You're asking for a Dockerfile again!"* — confused by bloated history. Memory Router's optimized prompt got a clean answer immediately. This is the well-known "lost in the middle" effect.

---

## Memory System Architecture

### Four Memory Types

| Type | Purpose | Example |
|---|---|---|
| **Semantic** | Durable facts | "User's stack is TypeScript + pnpm" |
| **Episodic** | Past events/interactions | "User debugged a CORS issue on 2024-03-15" |
| **Procedural** | How-to knowledge | "Deploy to prod: run tests → build → push → tag" |
| **Working** | Session-scoped scratch | Current file being edited, active error message |

### Mycelium Network

Inspired by fungal mycelium networks, memories form associative connections:

- **Co-retrieval strengthening**: When memories are retrieved together, the edge between them gets stronger
- **Spreading activation**: Querying one memory surfaces associated memories through multi-hop graph traversal
- **Decay**: Unused edges weaken over time; very weak edges are pruned

### Confidence Decay

Memories lose confidence over time unless reinforced:

- **Decay**: Exponential decay of *confidence* (temporal reliability) based on days since last use — importance (your weight) is never touched
- **Reinforcement**: Using a memory boosts its confidence and resets the decay clock
- **Pruning**: `memory-router memory decay --prune` removes memories whose confidence falls below threshold

### Memory Consolidation

Over time, similar memories accumulate. Consolidation merges near-duplicates:

```bash
memory-router memory consolidate          # preview (dry run)
memory-router memory consolidate --apply  # merge clusters
```

The algorithm uses Jaccard word-overlap with union-find clustering. The highest-importance memory becomes the anchor; concepts are merged from all cluster members.

---

## Project Layout

```
memory_router/
├── __init__.py                # Public API exports (v0.2)
├── cli.py                     # Typer CLI with streaming, import/export, routing
├── config.py                  # Config with validation (enum, range constraints)
├── classifier.py              # Rule-based task/domain/concept extraction
├── context_builder.py         # Priority-scored context assembly
├── token_optimizer.py         # Budget enforcement (priority + positional)
├── ask_service.py             # Shared classify→route→complete orchestration
├── router.py                  # Rule-based routing with fallback + retry
├── adaptive_router.py         # Outcome-learning adaptive router
├── mcp_server.py              # MCP server (20 tools, rate-limited, sanitized)
├── health.py                  # Structured health checks
├── stats.py                   # Cumulative usage statistics
├── benchmark.py               # Token savings & quality benchmarks
├── memory/
│   ├── sqlite_store.py        # MemoryStore + ConversationStore + FTS5
│   ├── palace.py              # Domain → task hierarchy view
│   ├── mycelium.py            # Associative memory graph
│   ├── working_memory.py      # Session-scoped scratchpad
│   ├── decay.py               # Confidence decay + reinforcement
│   ├── consolidation.py       # Near-duplicate merging
│   ├── importer.py            # Import/export (ChatGPT, Claude, generic JSON)
│   ├── summarizer.py          # Sentence-boundary summarizer
│   ├── vector_store.py        # NumPy/pure-Python cosine similarity backend
│   ├── auto_capture.py        # Auto-promote useful turns to memory
│   └── obsidian/              # Opt-in Obsidian export layer (one-way projection)
│       ├── vault.py           #   safe writes, backups, vault scaffold
│       ├── exporter.py        #   knowledge/raw/project export, idempotent
│       ├── renderer.py        #   Markdown render + lossless round-trip
│       ├── index.py           #   content-hash manifest for incremental export
│       ├── models.py          #   KnowledgeNote, ExportResult, folder mapping
│       └── utils.py           #   slugify, safe paths, ISO time, redaction
├── providers/
│   ├── base.py                # BaseProvider + StreamChunk interfaces
│   ├── ollama_provider.py     # Local Ollama via HTTP (streaming)
│   ├── openai_provider.py     # OpenAI SDK (streaming)
│   ├── anthropic_provider.py  # Anthropic SDK (streaming)
│   ├── gemini_provider.py     # Google GenAI SDK (streaming)
│   └── ruflo_provider.py      # Optional multi-agent backend
├── security/
│   ├── keychain.py            # OS keychain + HMAC-protected fallback
│   └── encryption.py          # AES-256-GCM with machine-derived keys
└── utils/
    ├── tokens.py              # tiktoken + heuristic estimator
    └── logging.py             # JSON-structured rotating file logs
```

---

## Python API

```python
from memory_router import (
    MemoryStore, Memory, classify, Config,
    build_context, Router, check_health, load_config,
)

# Store and search memories
store = MemoryStore()
store.add(Memory(content="User prefers dark mode", domain="prefs", importance=0.9))
results = store.search(query_text="theme preferences")

# Find near-duplicates before storing
similar = store.find_similar("User likes dark theme", threshold=0.6)

# Classify a query
cls = classify("Write a Python function to sort a list")
# → Classification(task='code', domain='software', concepts=[...], complexity=0.6)

# Health check
report = check_health()
print(report.overall)  # "ok" | "degraded" | "unhealthy"
```

---

## Security

- **API keys**: OS keychain first, HMAC-SHA256 verified fallback file at 0600 permissions
- **Encrypted export/import**: Optional AES-256-GCM for exported/imported memory files with machine-derived keys (`pip install memory-router[encryption]`). The live SQLite databases are stored unencrypted — encryption applies to portable export files, not the working DB.
- **Input validation**: All MCP tool inputs sanitized (session IDs regex-validated, text truncated at boundaries)
- **Rate limiting**: Configurable per-minute limit on MCP tool calls (default: 100/min)
- **No secrets in config**: Keys never written to `config.yaml`
- **Tamper detection**: HMAC integrity check on the secrets fallback file

---

## What Memory Router is *not*

- ❌ Not a coding assistant or pair-programmer
- ❌ Does not execute shell commands
- ❌ Does not modify your files or run git
- ❌ Not an IDE or Claude Code replacement
- ✅ A **memory + prompt-optimization layer** that works alongside your existing tools

---

## Roadmap

- [x] ~~FTS5 full-text search~~
- [x] ~~Streaming responses~~
- [x] ~~Import from ChatGPT/Claude exports~~
- [x] ~~tiktoken integration~~
- [x] ~~Per-domain importance decay~~
- [x] ~~Adaptive routing with outcome learning~~
- [x] ~~Mycelium associative memory network~~
- [x] ~~Encrypted export/import~~
- [x] ~~Vector similarity backend (NumPy / pure-Python cosine)~~
- [x] ~~Obsidian export layer with mycelium-backed Graph View~~
- [ ] Auto-generated embeddings + ANN index (FAISS / sqlite-vec / Chroma)
- [ ] LLM-backed concept extraction as a classifier fallback
- [ ] LLM-based summarizer (opt-in, runs on local model)
- [ ] Multi-user / team memory sharing
- [ ] Web dashboard for memory visualization
- [ ] First-class Ruflo plugin once the public API stabilizes

---

## License

MIT.
