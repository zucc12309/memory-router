"""memory-router CLI.

This is the user-facing surface. Everything else is library code and could be
embedded in another app. Built with Typer for clean subcommands and Rich for
readable output.
"""

from __future__ import annotations

import json
import sys
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


def _explain_provider_error(provider_name: str, model: str, err: Exception) -> None:
    """Render a friendly, actionable error for provider failures.

    We pattern-match on common cases (Ollama not running, missing SDK, missing
    API key, auth failure, model not found) and surface a remediation hint.
    Falls back to the raw error so debugging info is never hidden.
    """
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
):
    """Ask a question — builds context, routes to a model, returns the answer."""
    _ask(query=query, no_memory=no_memory, local=local, session=session,
         override_provider=provider, override_model=model)


def _ask(query: str, no_memory: bool, local: bool, session: str,
         override_provider: Optional[str] = None, override_model: Optional[str] = None):
    cfg = _require_init()

    # Stage 1: classify + build context. Any failure here is local-only and
    # almost always indicates a corrupted SQLite file or bad config.
    try:
        classification = classify(query)
        mem_store = MemoryStore()
        conv_store = ConversationStore()
        built = build_context(
            query=query,
            classification=classification,
            cfg=cfg,
            mem_store=mem_store,
            conv_store=conv_store,
            use_memory=not no_memory,
            session_id=session,
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

    # Stage 2: routing. Should never fail, but be defensive.
    try:
        router = Router(cfg)
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

    # Stage 3: provider call — most likely failure point.
    try:
        result = decision.provider.complete(decision.model, built.messages)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        _explain_provider_error(decision.provider.name, decision.model, e)
        raise typer.Exit(code=2)

    console.print(Panel(result.text or "[no response]", title="Answer", border_style="green"))

    # Real token usage + cost. Uses the SDK's reported numbers when available
    # (real billed tokens), the estimator otherwise. The "saved" line compares
    # the optimized input against what a naive full-history send would have used.
    real_in = result.input_tokens or built.sent_tokens
    real_out = result.output_tokens or 0
    naive_in = built.full_history_tokens or built.sent_tokens
    real_saved = percent_saved(naive_in, real_in)
    cost = estimate_cost_usd(result.model, real_in, real_out)

    token_table = Table.grid(padding=(0, 2))
    token_table.add_column(style="cyan", justify="right")
    token_table.add_column()
    token_table.add_row("Input tokens (real):", f"{real_in:,}")
    token_table.add_row("Output tokens (real):", f"{real_out:,}")
    token_table.add_row("Naive baseline (est.):", f"~{naive_in:,}")
    token_table.add_row("Saved on input:", f"{real_saved}%")
    token_table.add_row("Cost (estimate):", format_cost(cost))
    console.print(Panel(token_table, title="Token usage", border_style="dim"))

    # Promote useful turns into long-term memory when the config allows it.
    try:
        memory_id = capture_turn(
            query=query,
            answer=result.text or "",
            classification=classification,
            cfg=cfg,
            store=mem_store,
            allow_capture=not no_memory,
        )
        if memory_id is not None:
            console.print(f"[dim]Auto-saved memory #{memory_id}.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: could not auto-save memory — {e}[/yellow]")

    # Log the turn locally — non-fatal if this fails (just warn).
    try:
        conv_store.add(Message(session_id=session, role="user", content=query))
        conv_store.add(Message(session_id=session, role="assistant", content=result.text or ""))
    except Exception as e:
        console.print(f"[yellow]Warning: could not save conversation turn — {e}[/yellow]")


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
    """Build an optimized prompt and print it — no LLM is called.

    Use this when you want to copy-paste an optimized, memory-augmented prompt
    into ChatGPT, Claude.ai, Claude Code, VS Code, or any other tool you
    already use. Memory Router stays a context layer; you keep your tools.
    """
    cfg = _require_init()
    try:
        classification = classify(query)
        mem_store = MemoryStore()
        conv_store = ConversationStore()
        built = build_context(
            query=query,
            classification=classification,
            cfg=cfg,
            mem_store=mem_store,
            conv_store=conv_store,
            use_memory=not no_memory,
            session_id=session,
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

    # Show which memories were pulled in.
    if built.used_memories:
        console.print("[bold cyan]Relevant memory used:[/bold cyan]")
        for m in built.used_memories:
            console.print(f"- [{m.domain}/{m.task}] {m.content}")
    else:
        console.print("[dim]Relevant memory used: (none)[/dim]")
    console.print()

    # Show classification + token savings so the user understands the trim.
    saved = percent_saved(built.full_history_tokens, built.sent_tokens)
    console.print(
        f"[dim]task={classification.task}  domain={classification.domain}  "
        f"concepts={classification.concepts}  tokens_sent≈{built.sent_tokens}  "
        f"saved≈{saved}%[/dim]"
    )
    console.print()

    if show_messages:
        # Role-tagged form, useful for tools that accept system+user separately.
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
    """Run a self-check: config, storage, and provider availability."""
    table = Table(title="Memory Router Doctor", show_lines=False)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    # 1. Initialized?
    if is_initialized():
        table.add_row("config file", "[green]ok[/green]", str(CONFIG_PATH))
        cfg = load_config()
    else:
        table.add_row("config file", "[red]missing[/red]", "Run: memory-router init")
        console.print(table)
        raise typer.Exit(code=1)

    # 2. Storage dirs + DBs.
    try:
        ensure_dirs()
        ms = MemoryStore()
        cs = ConversationStore()
        n_mem = len(ms.list_all(limit=1_000_000))
        n_msg = len(cs.all_for_session("default"))
        table.add_row("memory palace db", "[green]ok[/green]", f"{n_mem} memories")
        table.add_row("conversations db", "[green]ok[/green]", f"{n_msg} messages in default session")
    except Exception as e:
        table.add_row("storage", "[red]error[/red]", str(e))

    # 3. Providers — check each one's availability.
    try:
        router = Router(cfg)
        for name, p in router.providers.items():
            try:
                ok = p.is_available()
            except Exception as e:
                ok = False
                detail = f"error: {e}"
            else:
                detail = "ready" if ok else "not configured / unavailable"
            table.add_row(f"provider:{name}", "[green]ok[/green]" if ok else "[yellow]–[/yellow]", detail)
    except Exception as e:
        table.add_row("router", "[red]error[/red]", str(e))

    # 4. Mode summary.
    table.add_row("mode", "[green]ok[/green]", cfg.mode)
    table.add_row(
        "memory_enabled",
        "[green]ok[/green]" if cfg.memory_enabled else "[yellow]off[/yellow]",
        str(cfg.memory_enabled),
    )
    table.add_row(
        "auto_capture_memories",
        "[green]ok[/green]" if cfg.auto_capture_memories else "[yellow]off[/yellow]",
        str(cfg.auto_capture_memories),
    )

    console.print(table)


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
        summary.extend(
            [
                f"avg score baseline≈{report.baseline_score_avg:.2f}",
                f"avg score optimized≈{report.optimized_score_avg:.2f}",
                f"avg delta≈{report.quality_delta_avg:.2f}",
            ]
        )
    console.print("[bold]Summary:[/bold] " + "  ".join(summary))
    if report.note:
        console.print(f"[yellow]{report.note}[/yellow]")


# ---------- benchmark ----------

@app.command()
def benchmark(
    cases: Optional[Path] = typer.Option(
        None,
        "--cases",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional JSON file with benchmark cases.",
    ),
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


# ---------- memory subcommands ----------

@memory_app.command("list")
def memory_list(limit: int = typer.Option(20, help="Max rows to show.")):
    _require_init()
    store = MemoryStore()
    mems = store.list_all(limit=limit)
    if not mems:
        console.print("[yellow]No memories yet. Add one with `memory-router memory add ...`[/yellow]")
        return
    table = Table(title="Memories")
    for col in ("id", "domain", "task", "importance", "uses", "content"):
        table.add_column(col)
    for m in mems:
        content = m.content if len(m.content) < 80 else m.content[:77] + "..."
        table.add_row(str(m.id), m.domain, m.task, f"{m.importance:.2f}", str(m.usage_count), content)
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
):
    _require_init()
    store = MemoryStore()
    cs = [c.strip() for c in concepts.split(",")] if concepts else []
    mem_id = store.add(Memory(task=task, domain=domain, concepts=cs, content=content, importance=importance))
    console.print(f"[green]Added memory #{mem_id}.[/green]")


@memory_app.command("delete")
def memory_delete(memory_id: int = typer.Argument(...)):
    _require_init()
    store = MemoryStore()
    ok = store.delete(memory_id)
    console.print("[green]Deleted.[/green]" if ok else "[yellow]Not found.[/yellow]")


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


# Names that should be dispatched to subcommands as-is. Anything else that
# looks like a free-form query gets the implicit `ask` shorthand.
_KNOWN_COMMANDS = {
    "ask", "init", "auth", "config", "memory", "build-context", "doctor", "benchmark",
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
    """Console entry point.

    Lets users write `memory-router "Explain bond convexity"` as shorthand for
    `memory-router ask "Explain bond convexity"`. We only rewrite argv when the
    first non-flag token is clearly a free-form query (i.e. not a known
    subcommand) so real subcommands like `init` keep working.
    """
    import sys
    sys.argv = [sys.argv[0], *_rewrite_argv(sys.argv[1:])]
    app()


if __name__ == "__main__":
    entry()
