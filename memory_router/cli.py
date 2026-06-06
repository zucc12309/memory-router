"""memory-router CLI.

This is the user-facing surface. Everything else is library code and could be
embedded in another app. Built with Typer for clean subcommands and Rich for
readable output.

v2: Integrates adaptive routing, fallback, streaming, mycelium, working memory,
    memory decay, import/export, and new doctor checks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from . import __version__
from .benchmark import BenchmarkSummary, load_cases, run_suite
from .classifier import classify
from .config import (
    CONFIG_PATH,
    Config,
    ROOT_DIR,
    ensure_dirs,
    is_initialized,
    load_config,
    save_config,
    set_value,
)
from .context_builder import build_context
from .memory.auto_capture import capture_turn
from .memory.palace import build_palace
from .memory.sqlite_store import ConversationStore, Memory, MemoryStore, Message
from .router import Router
from .security.keychain import delete_secret, get_secret, set_secret
from .stats import record_usage, summarize_stats, reset_stats
from .utils.tokens import percent_saved, estimate_cost_usd, format_cost


app = typer.Typer(
    name="memory-router",
    help=(
        "Local-first context optimization layer with a structured Memory Palace. "
        "We don't replace LLMs — we optimize how they are used."
    ),
    no_args_is_help=False,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect or change config values.")
memory_app = typer.Typer(help="Manage the Memory Palace.")
app.add_typer(config_app, name="config")
app.add_typer(memory_app, name="memory")

console = Console()


# ---------- helpers ----------

def _require_init() -> Config:
    if not is_initialized():
        console.print("[yellow]No config found. Run [bold]memory-router init[/bold] first.[/yellow]")
        raise typer.Exit(code=1)
    return load_config()


def _get_router(cfg: Config):
    """Return the adaptive router when enabled, otherwise the rule-based one."""
    if cfg.adaptive_routing:
        from .adaptive_router import AdaptiveRouter
        return AdaptiveRouter(cfg)
    return Router(cfg)


def _get_mycelium(mem_store: MemoryStore, cfg: Config):
    """Return a MyceliumNetwork if enabled."""
    if cfg.mycelium_enabled:
        from .memory.mycelium import MyceliumNetwork
        return MyceliumNetwork(mem_store.conn)
    return None


def _apply_decay_if_enabled(mem_store: MemoryStore, cfg: Config) -> None:
    """Apply memory decay lazily on every query when enabled."""
    if cfg.memory_decay_enabled:
        try:
            from .memory.decay import apply_decay
            apply_decay(mem_store)
        except Exception:
            pass


def _explain_provider_error(provider_name: str, model: str, err: Exception) -> None:
    """Render a friendly, actionable error for provider failures."""
    msg = str(err)
    low = msg.lower()
    console.print(f"[red bold]Provider error[/red bold] ([cyan]{provider_name}[/cyan] / [cyan]{model}[/cyan])")
    console.print(f"[red]{msg}[/red]")

    hint = None
    if provider_name == "ollama" and ("connection refused" in low or "max retries" in low or "newconnectionerror" in low):
        hint = (
            "Ollama doesn't appear to be running.\n"
            "  • Install:  brew install ollama   (or download from https://ollama.com)\n"
            "  • Start:    ollama serve &\n"
            f"  • Pull:     ollama pull {model}\n"
            "  • Or skip the LLM call entirely with: memory-router build-context \"...\""
        )
    elif "no openai api key" in low or ("openai" in low and "api key" in low):
        hint = "Add an OpenAI key:  memory-router auth openai"
    elif "no anthropic api key" in low or ("anthropic" in low and "api key" in low):
        hint = "Add an Anthropic key:  memory-router auth anthropic"
    elif "no gemini api key" in low or ("gemini" in low and "api key" in low):
        hint = "Add a Gemini key:  memory-router auth gemini"
    elif "package not installed" in low or "no module named" in low:
        if "openai" in low:
            hint = "Install the OpenAI SDK:  pip install \"memory-router[openai]\""
        elif "anthropic" in low:
            hint = "Install the Anthropic SDK:  pip install \"memory-router[anthropic]\""
        elif "google" in low or "gemini" in low or "generativeai" in low:
            hint = "Install the Gemini SDK:  pip install \"memory-router[gemini]\""
        else:
            hint = "Install all optional providers:  pip install \"memory-router[all]\""
    elif "401" in msg or "unauthorized" in low or "invalid api key" in low or "authentication" in low:
        hint = (
            f"The {provider_name} API key looks invalid or expired.\n"
            f"  • Re-add:  memory-router auth {provider_name}\n"
            f"  • Remove:  memory-router auth {provider_name} --delete"
        )
    elif "404" in msg or "not found" in low or "does not exist" in low or "model_not_found" in low:
        hint = (
            f"The model id '{model}' isn't available on this account.\n"
            f"  • Edit ~/.memory-router/config.yaml under `models:` to a model your API key supports."
        )
    elif "429" in msg or "rate limit" in low or "quota" in low:
        hint = "Rate-limited or out of quota. Wait a moment, or switch providers via `memory-router config set mode hybrid`."
    elif "timeout" in low or "timed out" in low:
        hint = "Request timed out. Check your network, or try a smaller model."

    if hint:
        console.print(Panel(hint, title="What to try", border_style="yellow"))


def _print_routing_header(provider_name: str, model: str, reason: str, used_memories, est_saved_pct: int):
    if used_memories:
        m = used_memories[0]
        memory_path = f"{m.task.title()} > {m.domain.title()}"
        if m.concepts:
            memory_path += " > " + ", ".join(m.concepts[:2])
    else:
        memory_path = "(none)"
    console.print(f"[bold]Using:[/bold] {provider_name} / {model}")
    console.print(f"[bold]Route:[/bold] {reason}")
    console.print(f"[bold]Memory used:[/bold] {memory_path}")
    console.print(f"[bold]Estimated tokens saved:[/bold] {est_saved_pct}%")
    console.print()


# ---------- top-level commands ----------

@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
):
    """Local-first context optimization layer. We don't replace LLMs — we optimize how they are used."""
    if version:
        console.print(f"memory-router {__version__}")
        raise typer.Exit()


@app.command("ask")
def ask_cmd(
    query: str = typer.Argument(..., help="The question to ask."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip memory retrieval for this query."),
    local: bool = typer.Option(False, "--local", help="Force local model only."),
    session: str = typer.Option("default", "--session", help="Conversation session id."),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Pin a provider for this call: openai | anthropic | gemini | ollama."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Pin a specific model id (e.g. gemini-2.5-flash, gpt-4o-mini)."
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream the response token-by-token."),
):
    """Ask a question — builds context, routes to a model, returns the answer."""
    _ask(query=query, no_memory=no_memory, local=local, session=session,
         override_provider=provider, override_model=model, stream=stream)


def _ask(query: str, no_memory: bool, local: bool, session: str,
         override_provider: Optional[str] = None, override_model: Optional[str] = None,
         stream: bool = False):
    cfg = _require_init()

    # Stage 1: classify + build context.
    try:
        classification = classify(query)
        mem_store = MemoryStore()
        conv_store = ConversationStore()

        _apply_decay_if_enabled(mem_store, cfg)
        mycelium = _get_mycelium(mem_store, cfg)

        built = build_context(
            query=query,
            classification=classification,
            cfg=cfg,
            mem_store=mem_store,
            conv_store=conv_store,
            use_memory=not no_memory,
            session_id=session,
            mycelium=mycelium,
        )
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to build context:[/red] {e}\n\n"
            "This usually means a corrupted SQLite file under ~/.memory-router/.\n"
            "Try: [bold]memory-router memory clear --yes[/bold] or remove the directory and re-init.",
            title="Context build error",
            border_style="red",
        ))
        raise typer.Exit(code=2)

    # Stage 2: routing.
    try:
        router = _get_router(cfg)
        decision = router.route(
            classification,
            force_local=local,
            override_provider=override_provider,
            override_model=override_model,
        )
    except Exception as e:
        console.print(Panel(
            f"[red]Router failed:[/red] {e}\n\n"
            "Check `memory-router config show` and re-run `memory-router init` if needed.",
            title="Routing error",
            border_style="red",
        ))
        raise typer.Exit(code=2)

    saved = percent_saved(built.full_history_tokens, built.sent_tokens)
    _print_routing_header(decision.provider.name, decision.model, decision.reason, built.used_memories, saved)
    if cfg.mode == "hybrid" and decision.provider.name != "ollama":
        console.print("[yellow]Note: this prompt will be sent to a remote provider.[/yellow]")

    # Stage 3: provider call with fallback.
    t0 = time.time()
    actual_provider = decision.provider.name

    if stream:
        result_text, real_in, real_out = _stream_response(decision, built, router)
    else:
        try:
            if hasattr(router, "complete_with_fallback"):
                result, actual_provider = router.complete_with_fallback(decision, built.messages)
            else:
                result = decision.provider.complete(decision.model, built.messages)
            result_text = result.text or "[no response]"
            real_in = result.input_tokens or built.sent_tokens
            real_out = result.output_tokens or 0
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            raise typer.Exit(code=130)
        except Exception as e:
            _explain_provider_error(decision.provider.name, decision.model, e)
            # Record the failure for adaptive routing
            _record_adaptive_outcome(cfg, router, decision, classification, 0, 0,
                                     int((time.time() - t0) * 1000), 0, str(e))
            raise typer.Exit(code=2)

    latency_ms = int((time.time() - t0) * 1000)

    if not stream:
        console.print(Panel(result_text, title="Answer", border_style="green"))

    naive_in = built.full_history_tokens or built.sent_tokens
    real_saved = percent_saved(naive_in, real_in)
    cost = estimate_cost_usd(decision.model, real_in, real_out)

    token_table = Table.grid(padding=(0, 2))
    token_table.add_column(style="cyan", justify="right")
    token_table.add_column()
    token_table.add_row("Input tokens (real):", f"{real_in:,}")
    token_table.add_row("Output tokens (real):", f"{real_out:,}")
    token_table.add_row("Naive baseline (est.):", f"~{naive_in:,}")
    token_table.add_row("Saved on input:", f"{real_saved}%")
    token_table.add_row("Latency:", f"{latency_ms:,}ms")
    token_table.add_row("Cost (estimate):", format_cost(cost))
    if actual_provider != decision.provider.name:
        token_table.add_row("Fallback used:", f"{actual_provider}")
    console.print(Panel(token_table, title="Token usage", border_style="dim"))

    record_usage(
        kind="cli_ask",
        naive_tokens=naive_in,
        sent_tokens=real_in,
        output_tokens=real_out,
        memories_used=len(built.used_memories),
        provider=actual_provider,
        model=decision.model,
        cost_usd=cost,
    )

    # Record for adaptive routing
    _record_adaptive_outcome(cfg, router, decision, classification,
                             real_in, real_out, latency_ms, cost, None)

    # Auto-capture
    try:
        memory_id = capture_turn(
            query=query,
            answer=result_text if not stream else "",
            classification=classification,
            cfg=cfg,
            store=mem_store,
            allow_capture=not no_memory,
        )
        if memory_id is not None:
            console.print(f"[dim]Auto-saved memory #{memory_id}.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: could not auto-save memory — {e}[/yellow]")

    # Log the turn
    try:
        conv_store.add(Message(session_id=session, role="user", content=query))
        conv_store.add(Message(session_id=session, role="assistant", content=result_text if not stream else ""))
    except Exception as e:
        console.print(f"[yellow]Warning: could not save conversation turn — {e}[/yellow]")


def _stream_response(decision, built, router):
    """Stream the response to the console in real time."""
    console.print("[bold green]Answer:[/bold green]")
    full_text = []
    real_in, real_out = 0, 0

    try:
        for chunk in decision.provider.stream(decision.model, built.messages):
            if chunk.text:
                console.print(chunk.text, end="")
                full_text.append(chunk.text)
            if chunk.finished:
                real_in = chunk.input_tokens or built.sent_tokens
                real_out = chunk.output_tokens or 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=130)
    except Exception:
        # Fall back to non-streaming
        result = decision.provider.complete(decision.model, built.messages)
        console.print(result.text or "[no response]")
        return result.text or "[no response]", result.input_tokens or built.sent_tokens, result.output_tokens or 0

    console.print()  # newline after stream
    combined = "".join(full_text)
    if not real_in:
        real_in = built.sent_tokens
    return combined, real_in, real_out


def _record_adaptive_outcome(cfg, router, decision, classification,
                              input_tokens, output_tokens, latency_ms, cost, error):
    """Record outcome for adaptive routing if enabled."""
    if not cfg.adaptive_routing:
        return
    try:
        from .adaptive_router import AdaptiveRouter, RouteOutcome
        if isinstance(router, AdaptiveRouter):
            # Estimate quality from auto-capture success
            quality = 0.7 if error is None else 0.1
            router.record_outcome(RouteOutcome(
                provider=decision.provider.name,
                model=decision.model,
                task=classification.task,
                domain=classification.domain,
                complexity=classification.complexity,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                quality_signal=quality,
                error=error,
            ))
    except Exception:
        pass


# ---------- build-context (no LLM call) ----------

@app.command("build-context")
def build_context_cmd(
    query: str = typer.Argument(..., help="The question/prompt you want to optimize."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip Memory Palace retrieval."),
    session: str = typer.Option("default", "--session", help="Conversation session id."),
    show_messages: bool = typer.Option(
        False, "--show-messages", help="Print the full role-tagged message list instead of a flat prompt."
    ),
):
    """Build an optimized prompt and print it — no LLM is called."""
    cfg = _require_init()
    try:
        classification = classify(query)
        mem_store = MemoryStore()
        conv_store = ConversationStore()
        _apply_decay_if_enabled(mem_store, cfg)
        mycelium = _get_mycelium(mem_store, cfg)

        built = build_context(
            query=query,
            classification=classification,
            cfg=cfg,
            mem_store=mem_store,
            conv_store=conv_store,
            use_memory=not no_memory,
            session_id=session,
            mycelium=mycelium,
        )
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to build context:[/red] {e}\n\n"
            "This usually means a corrupted SQLite file under ~/.memory-router/.\n"
            "Try: [bold]memory-router memory clear --yes[/bold] or remove the directory and re-init.",
            title="Context build error",
            border_style="red",
        ))
        raise typer.Exit(code=2)

    if built.used_memories:
        console.print("[bold cyan]Relevant memory used:[/bold cyan]")
        for m in built.used_memories:
            console.print(f"- [{m.domain}/{m.task}] {m.content}")
    else:
        console.print("[dim]Relevant memory used: (none)[/dim]")
    console.print()

    saved = percent_saved(built.full_history_tokens, built.sent_tokens)
    console.print(
        f"[dim]task={classification.task}  domain={classification.domain}  "
        f"concepts={classification.concepts}  tokens_sent≈{built.sent_tokens}  "
        f"saved≈{saved}%[/dim]"
    )
    console.print()

    if show_messages:
        console.print("[bold green]Optimized messages:[/bold green]")
        for m in built.messages:
            console.print(Panel(m["content"], title=m["role"], border_style="green"))
        return

    optimized_prompt = _render_flat_prompt(built.messages)
    console.print("[bold green]Optimized prompt:[/bold green]")
    console.print(optimized_prompt)


# ---------- init ----------

@app.command()
def init():
    """Interactive first-time setup."""
    ensure_dirs()
    console.print(Panel.fit("[bold]Memory Router setup[/bold]\nEverything is stored locally under "
                            f"{ROOT_DIR}", border_style="cyan"))

    mode = Prompt.ask(
        "Choose mode",
        choices=["local", "api", "hybrid", "ruflo"],
        default="hybrid",
    )

    cfg = load_config()
    cfg.mode = mode

    cfg.memory_enabled = Confirm.ask("Enable Memory Palace?", default=True)
    cfg.auto_capture_memories = (
        Confirm.ask("Auto-save useful turns into Memory Palace?", default=True)
        if cfg.memory_enabled
        else False
    )

    if mode in ("api", "hybrid"):
        if Confirm.ask("Add an OpenAI API key now?", default=False):
            key = Prompt.ask("OpenAI API key", password=True)
            backend = set_secret("openai", key)
            console.print(f"[green]Saved OpenAI key to {backend}.[/green]")
        if Confirm.ask("Add an Anthropic API key now?", default=False):
            key = Prompt.ask("Anthropic API key", password=True)
            backend = set_secret("anthropic", key)
            console.print(f"[green]Saved Anthropic key to {backend}.[/green]")
        if Confirm.ask("Add a Google Gemini API key now?", default=False):
            key = Prompt.ask("Gemini API key", password=True)
            backend = set_secret("gemini", key)
            console.print(f"[green]Saved Gemini key to {backend}.[/green]")

    if mode in ("local", "hybrid"):
        host = Prompt.ask("Ollama host", default=cfg.ollama_host)
        cfg.ollama_host = host

    # v2 features
    cfg.mycelium_enabled = Confirm.ask("Enable mycelium memory network?", default=True)
    cfg.memory_decay_enabled = Confirm.ask("Enable memory decay (stale memories auto-fade)?", default=True)

    save_config(cfg)
    console.print(f"[green]Wrote config to {CONFIG_PATH}.[/green]")
    console.print("Try: [bold]memory-router \"Explain bond convexity\"[/bold]")


# ---------- auth ----------

@app.command()
def auth(
    provider: str = typer.Argument(..., help="Provider name: openai | anthropic | gemini | ..."),
    delete: bool = typer.Option(False, "--delete", help="Remove the saved credential."),
):
    """Store an API key in the OS keychain (or a 0600-permission fallback file)."""
    if delete:
        ok = delete_secret(provider)
        console.print("[green]Deleted.[/green]" if ok else "[yellow]No credential to delete.[/yellow]")
        return
    key = Prompt.ask(f"{provider} API key", password=True)
    backend = set_secret(provider, key)
    console.print(f"[green]Saved {provider} credential via {backend}.[/green]")


# ---------- config subcommands ----------

@config_app.command("show")
def config_show():
    """Print the current config."""
    cfg = _require_init()
    table = Table(title="Memory Router Config", show_lines=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for k, v in cfg.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key, e.g. mode, token_budget."),
    value: str = typer.Argument(..., help="New value."),
):
    """Update a single config field."""
    _require_init()
    try:
        cfg = set_value(key, value)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Set {key} = {getattr(cfg, key)}.[/green]")


# ---------- doctor ----------

@app.command()
def doctor():
    """Run a self-check: config, storage, providers, and v2 subsystems."""
    table = Table(title="Memory Router Doctor", show_lines=False)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    if is_initialized():
        table.add_row("config file", "[green]ok[/green]", str(CONFIG_PATH))
        cfg = load_config()
    else:
        table.add_row("config file", "[red]missing[/red]", "Run: memory-router init")
        console.print(table)
        raise typer.Exit(code=1)

    # Storage dirs + DBs
    try:
        ensure_dirs()
        ms = MemoryStore()
        cs = ConversationStore()
        n_mem = ms.count()
        n_msg = cs.session_count("default")
        table.add_row("memory palace db", "[green]ok[/green]", f"{n_mem} memories")
        table.add_row("conversations db", "[green]ok[/green]", f"{n_msg} messages in default session")
    except Exception as e:
        table.add_row("storage", "[red]error[/red]", str(e))

    # Providers
    try:
        router = Router(cfg)
        for name, p in router.providers.items():
            try:
                ok = p.is_available()
            except Exception:
                ok = False
            detail = "ready" if ok else "not configured / unavailable"
            table.add_row(f"provider:{name}", "[green]ok[/green]" if ok else "[yellow]–[/yellow]", detail)
    except Exception as e:
        table.add_row("router", "[red]error[/red]", str(e))

    # Mode + core config
    table.add_row("mode", "[green]ok[/green]", cfg.mode)
    table.add_row("memory_enabled", "[green]ok[/green]" if cfg.memory_enabled else "[yellow]off[/yellow]", str(cfg.memory_enabled))
    table.add_row("auto_capture", "[green]ok[/green]" if cfg.auto_capture_memories else "[yellow]off[/yellow]", str(cfg.auto_capture_memories))

    # v2 subsystems
    table.add_row("mycelium_network", "[green]on[/green]" if cfg.mycelium_enabled else "[yellow]off[/yellow]",
                  "associative memory graph" if cfg.mycelium_enabled else "disabled")
    table.add_row("memory_decay", "[green]on[/green]" if cfg.memory_decay_enabled else "[yellow]off[/yellow]",
                  "confidence decay enabled" if cfg.memory_decay_enabled else "disabled")
    table.add_row("adaptive_routing", "[green]on[/green]" if cfg.adaptive_routing else "[yellow]off[/yellow]",
                  "outcome-learning router" if cfg.adaptive_routing else "rule-based")

    # FTS5 check
    try:
        ms = MemoryStore()
        if ms._fts_available:
            table.add_row("fts5_search", "[green]ok[/green]", "full-text search active")
        else:
            table.add_row("fts5_search", "[yellow]–[/yellow]", "FTS5 not available in this SQLite build")
    except Exception:
        table.add_row("fts5_search", "[yellow]–[/yellow]", "could not check")

    # Tiktoken check
    try:
        import tiktoken  # noqa: F401
        table.add_row("tiktoken", "[green]ok[/green]", "precise token counting active")
    except ImportError:
        table.add_row("tiktoken", "[yellow]–[/yellow]", "using heuristic (pip install tiktoken)")

    # Encryption check
    try:
        from .security.encryption import is_encryption_available
        if is_encryption_available():
            table.add_row("encryption", "[green]ok[/green]" if cfg.encryption_enabled else "[yellow]available[/yellow]",
                          "AES-256-GCM ready" + ("" if cfg.encryption_enabled else " (enable: config set encryption_enabled true)"))
        else:
            table.add_row("encryption", "[yellow]–[/yellow]", "pip install cryptography")
    except Exception:
        table.add_row("encryption", "[yellow]–[/yellow]", "could not check")

    # Mycelium stats
    if cfg.mycelium_enabled:
        try:
            ms = MemoryStore()
            mycelium = _get_mycelium(ms, cfg)
            if mycelium:
                stats = mycelium.stats()
                table.add_row("mycelium_edges", "[green]ok[/green]", f"{stats['edge_count']} edges, avg weight {stats['avg_weight']}")
        except Exception:
            pass

    # Memory decay stats
    if cfg.memory_decay_enabled:
        try:
            ms = MemoryStore()
            from .memory.decay import get_decay_stats
            ds = get_decay_stats(ms)
            table.add_row("memory_health", "[green]ok[/green]",
                          f"{ds['strong_count']} strong, {ds['stale_count']} stale, avg importance {ds['avg_importance']:.2f}")
        except Exception:
            pass

    console.print(table)


# ---------- benchmark ----------

def _format_score(score: Optional[float]) -> str:
    return "n/a" if score is None else f"{score:.2f}"

def _format_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.2f}"

def _render_benchmark_report(report: BenchmarkSummary) -> None:
    table = Table(title="Memory Router Benchmark", show_lines=False)
    table.add_column("Case", style="cyan")
    table.add_column("Raw", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Optimized", justify="right")
    table.add_column("Saved", justify="right")
    table.add_column("Score B", justify="right")
    table.add_column("Score O", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status", style="yellow")

    for record in report.cases:
        delta = None
        if record.baseline_score is not None and record.optimized_score is not None:
            delta = record.optimized_score - record.baseline_score
        table.add_row(
            record.name,
            str(record.raw_tokens),
            str(record.baseline_tokens),
            str(record.optimized_tokens),
            f"{record.raw_saved_pct}%",
            _format_score(record.baseline_score),
            _format_score(record.optimized_score),
            _format_delta(delta),
            record.status if record.note else "ok",
        )

    console.print(table)

    summary = [
        f"avg raw≈{report.raw_tokens_avg:.0f}",
        f"avg baseline≈{report.baseline_tokens_avg:.0f}",
        f"avg optimized≈{report.optimized_tokens_avg:.0f}",
        f"avg saved≈{report.raw_saved_pct_avg:.1f}%",
    ]
    if report.baseline_score_avg is not None and report.optimized_score_avg is not None:
        summary.extend([
            f"avg score baseline≈{report.baseline_score_avg:.2f}",
            f"avg score optimized≈{report.optimized_score_avg:.2f}",
            f"avg delta≈{report.quality_delta_avg:.2f}",
        ])
    console.print("[bold]Summary:[/bold] " + "  ".join(summary))
    if report.note:
        console.print(f"[yellow]{report.note}[/yellow]")


@app.command()
def benchmark(
    cases: Optional[Path] = typer.Option(None, "--cases", exists=True, dir_okay=False, readable=True,
                                         help="Optional JSON file with benchmark cases."),
    no_run: bool = typer.Option(False, "--no-run", help="Skip model calls and only compare prompts."),
    local: bool = typer.Option(False, "--local", help="Force local routing when a model run is requested."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead of a table."),
):
    """Compare a naive prompt against the optimized Memory Router prompt."""
    cfg = load_config() if is_initialized() else Config()
    if not is_initialized():
        console.print("[yellow]No config found; using default benchmark settings.[/yellow]")

    try:
        suite = load_cases(cases)
        report = run_suite(suite, cfg=cfg, run_model=not no_run, force_local=local)
    except Exception as e:
        console.print(Panel(f"[red]Benchmark failed:[/red] {e}", title="Benchmark error", border_style="red"))
        raise typer.Exit(code=2)

    if json_output:
        console.print(json.dumps(report.to_dict(), indent=2, sort_keys=False))
        return

    _render_benchmark_report(report)


# ---------- mcp server ----------

mcp_app = typer.Typer(help="Run Memory Router as an MCP server for Claude Code, Cursor, etc.")
app.add_typer(mcp_app, name="mcp")

@mcp_app.command("serve")
def mcp_serve():
    """Start the MCP server on stdio."""
    _require_init()
    try:
        from .mcp_server import main as mcp_main
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    mcp_main()


# ---------- stats ----------

@app.command()
def stats(
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON."),
    reset: bool = typer.Option(False, "--reset", help="Wipe all stats and exit."),
):
    """Show cumulative token-saving stats across all CLI + MCP usage."""
    _require_init()

    if reset:
        if not Confirm.ask("Reset ALL stats? This cannot be undone.", default=False):
            console.print("Cancelled.")
            return
        n = reset_stats()
        console.print(f"[green]Reset {n} usage events.[/green]")
        return

    s = summarize_stats()
    if json_output:
        console.print(json.dumps(s.to_dict(), indent=2))
        return

    if s.calls == 0:
        console.print("[yellow]No usage recorded yet.[/yellow] Run a query first: memory-router \"hi\"")
        return

    headline = Table.grid(padding=(0, 2))
    headline.add_column(style="cyan", justify="right")
    headline.add_column()
    headline.add_row("Calls tracked:", f"{s.calls:,}")
    headline.add_row("Tokens that would have been sent:", f"{s.naive_tokens:,}")
    headline.add_row("Tokens actually sent:", f"{s.sent_tokens:,}")
    headline.add_row("Tokens saved:", f"[bold green]{s.tokens_saved:,}[/bold green]  ({s.saved_pct}%)")
    headline.add_row("Output tokens received:", f"{s.output_tokens:,}")
    headline.add_row("Memories injected:", f"{s.memories_used:,}")
    headline.add_row("Estimated cost (real):", format_cost(s.cost_usd))
    console.print(Panel(headline, title="Memory Router — Cumulative Savings", border_style="green"))

    if s.by_provider:
        prov_table = Table(title="By provider", show_header=True, show_lines=False)
        prov_table.add_column("Provider", style="cyan")
        prov_table.add_column("Calls", justify="right")
        prov_table.add_column("Naive", justify="right")
        prov_table.add_column("Sent", justify="right")
        prov_table.add_column("Cost", justify="right")
        for prov, p in sorted(s.by_provider.items()):
            prov_table.add_row(prov, f"{p['calls']:,}", f"{p['naive_tokens']:,}",
                               f"{p['sent_tokens']:,}", format_cost(p['cost_usd']))
        console.print(prov_table)

    if s.by_kind:
        kind_table = Table(title="By kind", show_header=True, show_lines=False)
        kind_table.add_column("Kind", style="cyan")
        kind_table.add_column("Calls", justify="right")
        kind_table.add_column("Naive", justify="right")
        kind_table.add_column("Sent", justify="right")
        for kind, k in sorted(s.by_kind.items()):
            kind_table.add_row(kind, f"{k['calls']:,}", f"{k['naive_tokens']:,}", f"{k['sent_tokens']:,}")
        console.print(kind_table)


# ---------- memory subcommands ----------

@memory_app.command("list")
def memory_list(
    limit: int = typer.Option(20, help="Max rows to show."),
    memory_type: Optional[str] = typer.Option(None, "--type", help="Filter by type: semantic|episodic|procedural."),
):
    _require_init()
    store = MemoryStore()
    if memory_type:
        mems = store.search_by_type(memory_type, limit=limit)
    else:
        mems = store.list_all(limit=limit)
    if not mems:
        console.print("[yellow]No memories yet. Add one with `memory-router memory add ...`[/yellow]")
        return
    table = Table(title="Memories")
    for col in ("id", "domain", "task", "type", "imp.", "conf.", "uses", "content"):
        table.add_column(col)
    for m in mems:
        content = m.content if len(m.content) < 70 else m.content[:67] + "..."
        table.add_row(str(m.id), m.domain, m.task, m.memory_type,
                      f"{m.importance:.2f}", f"{m.confidence:.2f}",
                      str(m.usage_count), content)
    console.print(table)


@memory_app.command("palace")
def memory_palace():
    """Show memories grouped by domain → task."""
    _require_init()
    store = MemoryStore()
    nodes = build_palace(store)
    if not nodes:
        console.print("[yellow]Memory Palace is empty.[/yellow]")
        return
    tree = Tree("[bold]Memory Palace[/bold]")
    for node in nodes:
        domain_branch = tree.add(f"[cyan]{node.domain}[/cyan]")
        for task_name, mems in node.tasks.items():
            task_branch = domain_branch.add(f"[magenta]{task_name}[/magenta] ({len(mems)})")
            for m in mems:
                snippet = m.content[:70] + ("..." if len(m.content) > 70 else "")
                task_branch.add(f"#{m.id} [{m.importance:.2f}] {snippet}")
    console.print(tree)


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="Memory content."),
    task: str = typer.Option("general", help="Task label."),
    domain: str = typer.Option("general", help="Domain label."),
    importance: float = typer.Option(0.5, help="0.0–1.0 importance score."),
    concepts: Optional[str] = typer.Option(None, help="Comma-separated concepts."),
    memory_type: str = typer.Option("semantic", "--type", help="Memory type: semantic|episodic|procedural."),
):
    _require_init()
    store = MemoryStore()
    cs = [c.strip() for c in concepts.split(",")] if concepts else []
    mem_id = store.add(Memory(task=task, domain=domain, concepts=cs, content=content,
                              importance=importance, memory_type=memory_type))
    console.print(f"[green]Added memory #{mem_id} (type={memory_type}).[/green]")


@memory_app.command("delete")
def memory_delete(memory_id: int = typer.Argument(...)):
    _require_init()
    store = MemoryStore()
    cfg = load_config()
    ok = store.delete(memory_id)
    if ok:
        # Clean up mycelium edges
        mycelium = _get_mycelium(store, cfg)
        if mycelium:
            mycelium.remove_memory(memory_id)
        console.print("[green]Deleted.[/green]")
    else:
        console.print("[yellow]Not found.[/yellow]")


@memory_app.command("clear")
def memory_clear(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
):
    _require_init()
    if not yes and not Confirm.ask("Delete ALL memories? This cannot be undone.", default=False):
        console.print("Cancelled.")
        return
    store = MemoryStore()
    n = store.clear()
    console.print(f"[green]Cleared {n} memories.[/green]")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, help="Max results."),
):
    """Search memory palace by query text using FTS5 + keyword scoring."""
    _require_init()
    store = MemoryStore()
    classification = classify(query)
    mems = store.search(
        task=classification.task,
        domain=classification.domain,
        concepts=classification.concepts,
        query_text=query,
        limit=limit,
    )
    if not mems:
        console.print("[yellow]No matching memories found.[/yellow]")
        return
    for m in mems:
        console.print(f"  [cyan]#{m.id}[/cyan] [{m.domain}/{m.task}] (imp={m.importance:.2f}, conf={m.confidence:.2f}) {m.content}")


@memory_app.command("decay")
def memory_decay_cmd(
    prune: bool = typer.Option(False, "--prune", help="Also prune memories below 0.05 importance."),
):
    """Apply memory decay and optionally prune stale memories."""
    _require_init()
    store = MemoryStore()
    from .memory.decay import apply_decay, prune_stale_memories, get_decay_stats

    decayed = apply_decay(store)
    console.print(f"[green]Decayed {decayed} memories.[/green]")

    if prune:
        pruned = prune_stale_memories(store)
        console.print(f"[green]Pruned {pruned} stale memories.[/green]")

    stats = get_decay_stats(store)
    console.print(f"[dim]Total: {stats['total_memories']} | "
                  f"Strong (≥0.8): {stats['strong_count']} | "
                  f"Stale (<0.1): {stats['stale_count']} | "
                  f"Avg importance: {stats['avg_importance']:.3f}[/dim]")


@memory_app.command("export")
def memory_export(
    output: Path = typer.Argument(..., help="Output JSON file path."),
):
    """Export all memories to a JSON file."""
    _require_init()
    store = MemoryStore()
    from .memory.importer import export_to_file
    n = export_to_file(store, output)
    console.print(f"[green]Exported {n} memories to {output}.[/green]")


@memory_app.command("import")
def memory_import(
    input_file: Path = typer.Argument(..., help="JSON file to import (Memory Router, ChatGPT, Claude, or generic)."),
):
    """Import memories from a JSON file. Auto-detects format."""
    _require_init()
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(code=1)

    store = MemoryStore()
    from .memory.importer import import_from_file
    imported, skipped = import_from_file(store, input_file)
    console.print(f"[green]Imported {imported} memories, skipped {skipped} duplicates.[/green]")


@memory_app.command("network")
def memory_network():
    """Show mycelium network statistics."""
    cfg = _require_init()
    if not cfg.mycelium_enabled:
        console.print("[yellow]Mycelium network is disabled. Enable: memory-router config set mycelium_enabled true[/yellow]")
        return

    store = MemoryStore()
    mycelium = _get_mycelium(store, cfg)
    if not mycelium:
        console.print("[yellow]Mycelium network not available.[/yellow]")
        return

    stats = mycelium.stats()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_row("Edges:", f"{stats['edge_count']:,}")
    table.add_row("Avg weight:", f"{stats['avg_weight']:.3f}")
    table.add_row("Max weight:", f"{stats['max_weight']:.3f}")
    table.add_row("Connected nodes:", f"{stats['connected_nodes']:,}")
    console.print(Panel(table, title="Mycelium Network", border_style="cyan"))


@memory_app.command("consolidate")
def memory_consolidate(
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Similarity threshold (0.0-1.0)."),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview without changes (default) or apply."),
):
    """Find and merge near-duplicate memories."""
    _require_init()
    store = MemoryStore()
    from .memory.consolidation import consolidate_memories

    result = consolidate_memories(store, similarity_threshold=threshold, dry_run=dry_run)

    if result.clusters_found == 0:
        console.print("[green]No near-duplicate memories found.[/green]")
        return

    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]APPLIED[/green]"
    console.print(f"\n{mode} — Consolidation results:")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_row("Clusters found:", str(result.clusters_found))
    table.add_row("Memories merged:", str(result.memories_merged))
    table.add_row("Memories remaining:", str(result.memories_remaining))
    console.print(Panel(table, title="Memory Consolidation", border_style="cyan"))

    if dry_run:
        console.print("\n[dim]Run with --apply to merge duplicates.[/dim]")


@memory_app.command("similar")
def memory_similar(
    text: str = typer.Argument(..., help="Text to find similar memories for."),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Similarity threshold."),
):
    """Find memories similar to given text."""
    _require_init()
    store = MemoryStore()
    similar = store.find_similar(text, threshold=threshold)

    if not similar:
        console.print("[yellow]No similar memories found.[/yellow]")
        return

    table = Table(title=f"Similar Memories (threshold={threshold})", show_lines=False)
    table.add_column("ID", style="dim", justify="right")
    table.add_column("Content")
    table.add_column("Importance", justify="right")

    for m in similar:
        table.add_row(str(m.id), m.content[:80], f"{m.importance:.2f}")
    console.print(table)


# ---------- routing report ----------

@app.command("routing-report")
def routing_report(
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON."),
):
    """Show adaptive routing performance report (when adaptive routing is enabled)."""
    cfg = _require_init()
    if not cfg.adaptive_routing:
        console.print("[yellow]Adaptive routing is disabled. Enable: memory-router config set adaptive_routing true[/yellow]")
        return

    from .adaptive_router import AdaptiveRouter
    router = AdaptiveRouter(cfg)
    report = router.get_performance_report()

    if not report:
        console.print("[yellow]No routing history yet. Run some queries first.[/yellow]")
        return

    if json_output:
        console.print(json.dumps([{
            "provider": p.provider, "model": p.model,
            "avg_quality": round(p.avg_quality, 3),
            "avg_latency_ms": round(p.avg_latency_ms, 0),
            "avg_cost": round(p.avg_cost, 6),
            "sample_count": p.sample_count,
            "error_rate": round(p.error_rate, 3),
        } for p in report], indent=2))
        return

    table = Table(title="Adaptive Routing Report", show_lines=False)
    table.add_column("Provider", style="cyan")
    table.add_column("Model")
    table.add_column("Quality", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Errors", justify="right")

    for p in report:
        table.add_row(
            p.provider, p.model,
            f"{p.avg_quality:.2f}",
            f"{p.avg_latency_ms:.0f}ms",
            format_cost(p.avg_cost),
            str(p.sample_count),
            f"{p.error_rate:.0%}",
        )
    console.print(table)


# ---------- argv rewriting ----------

_KNOWN_COMMANDS = {
    "ask", "init", "auth", "config", "memory", "build-context", "doctor", "benchmark",
    "mcp", "stats", "routing-report",
    "--help", "-h", "--version",
}


def _rewrite_argv(argv: list[str]) -> list[str]:
    """Insert the implicit `ask` command when the first real token is a query."""
    first_pos_idx = next((i for i, a in enumerate(argv) if not a.startswith("-")), None)
    if first_pos_idx is not None and argv[first_pos_idx] not in _KNOWN_COMMANDS:
        return ["ask", *argv]
    return argv


def _render_flat_prompt(messages: list[dict]) -> str:
    """Render the assembled chat as a single copy/paste-friendly prompt."""
    parts = []
    for m in messages:
        role = m.get("role", "message")
        content = m.get("content", "")
        if role == "system":
            parts.append(content)
        else:
            parts.append(f"{role.title()}: {content}")
    return "\n\n---\n\n".join(parts)


def entry() -> None:
    """Console entry point."""
    import sys
    sys.argv = [sys.argv[0], *_rewrite_argv(sys.argv[1:])]
    app()


if __name__ == "__main__":
    entry()
