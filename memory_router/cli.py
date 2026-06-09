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
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
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
from .security.keychain import delete_secret, set_secret
from .stats import record_usage, summarize_stats, reset_stats
from .utils.ollama import ensure_ollama_model_available, ensure_ollama_running
from .utils.system import (
    OLLAMA_MODEL_OPTIONS,
    detect_system_specs,
    normalize_ollama_model_name,
    recommend_ollama_model,
)
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

def _truncate_text(text: str, limit: int = 88) -> str:
    """Compact a long string for one-line summaries."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _memory_preview(mem: Memory, content_limit: int = 88) -> str:
    """Render a short human-readable description of a memory row."""
    parts = [f"{mem.domain}/{mem.task}"]
    if mem.memory_type:
        parts.append(f"({mem.memory_type})")
    if mem.concepts:
        concepts = ", ".join(mem.concepts[:2])
        if len(mem.concepts) > 2:
            concepts += ", ..."
        parts.append(f"[{concepts}]")
    if mem.content:
        parts.append(f"- {_truncate_text(mem.content, content_limit)}")
    return " ".join(parts)


def _memory_summary(used_memories) -> str:
    """Summarize the retrieved memories without dumping the whole list."""
    if not used_memories:
        return "none"
    top = used_memories[0]
    parts = [f"{top.domain}/{top.task}"]
    if top.memory_type:
        parts.append(f"({top.memory_type})")
    if top.concepts:
        concepts = ", ".join(top.concepts[:2])
        if len(top.concepts) > 2:
            concepts += ", ..."
        parts.append(f"[{concepts}]")
    top_label = " ".join(parts)
    if len(used_memories) == 1:
        return f"1 memory used: {top_label}"
    return f"{len(used_memories)} memories used: {top_label} (+{len(used_memories) - 1} more)"


def _summary_grid(rows: list[tuple[str, str]]) -> Table:
    """Build a compact key/value grid for status cards."""
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column(style="white")
    for label, value in rows:
        grid.add_row(label, value)
    return grid


def _summary_panel(
    title: str,
    rows: list[tuple[str, str]],
    border_style: str = "cyan",
    subtitle: Optional[str] = None,
) -> Panel:
    """Wrap a summary grid in a rounded panel."""
    return Panel(
        _summary_grid(rows),
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _message_border_style(role: str) -> str:
    """Use distinct colors for system/user/assistant message cards."""
    return {
        "system": "blue",
        "user": "green",
        "assistant": "magenta",
    }.get(role, "cyan")


def _render_message_panel(message: dict, index: int) -> Panel:
    """Render one role-tagged message panel for build-context output."""
    role = message.get("role", "message")
    content = message.get("content", "")
    return Panel(
        content,
        title=f"{role.title()} #{index}",
        border_style=_message_border_style(role),
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _memory_table(memories, title: str = "Relevant memories") -> Table:
    """Render retrieved memories in a readable table."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
        row_styles=["none", "dim"],
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Domain / Task", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Imp.", justify="right", no_wrap=True)
    table.add_column("Conf.", justify="right", no_wrap=True)
    table.add_column("Uses", justify="right", no_wrap=True)
    table.add_column("Content", overflow="fold")
    for idx, m in enumerate(memories, 1):
        concepts = ""
        if m.concepts:
            concepts = " [" + ", ".join(m.concepts[:2])
            if len(m.concepts) > 2:
                concepts += ", ..."
            concepts += "]"
        table.add_row(
            str(idx),
            f"{m.domain}/{m.task}{concepts}",
            m.memory_type,
            f"{m.importance:.2f}",
            f"{m.confidence:.2f}",
            str(m.usage_count),
            _truncate_text(m.content, 120),
        )
    return table


def _mode_guide_panel() -> Panel:
    """Explain the available setup modes without changing the flow."""
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("local", "Ollama only. Best for fully offline use and private workflows. Starts in the background on first use and suggests a model from your specs.")
    table.add_row("api", "Cloud providers only. Best when you already have keys and want remote models.")
    table.add_row("hybrid", "Local first, cloud fallback when needed. Best overall balance.")
    table.add_row("ruflo", "Agentic routing path for advanced workflows.")
    return Panel(
        table,
        title="Mode guide",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _system_specs_panel(specs=None, recommendation=None) -> Panel:
    """Show detected machine specs and the recommended local model."""
    specs = specs or detect_system_specs()
    recommendation = recommendation or recommend_ollama_model(specs)

    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("OS", specs.os_name)
    table.add_row("CPU", f"{specs.cpu_count} cores")
    table.add_row("Architecture", specs.architecture)
    table.add_row("RAM", f"{specs.memory_gb:.1f} GB" if specs.memory_gb is not None else "unknown")
    table.add_row("Recommended", f"{recommendation.model} ({recommendation.tier})")
    table.add_row("Why", recommendation.reason)

    return Panel(
        table,
        title="Local model recommendation",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _local_model_options_table() -> Table:
    """Render the Ollama model shortlist used during setup."""
    table = Table(
        title="Ollama model shortlist",
        box=box.ROUNDED,
        header_style="bold cyan",
        row_styles=["none", "dim"],
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Min RAM", justify="right", no_wrap=True)
    table.add_column("Label", style="magenta", no_wrap=True)
    table.add_column("Description", overflow="fold")
    for option in OLLAMA_MODEL_OPTIONS:
        table.add_row(
            option.model,
            f"{option.min_memory_gb:.0f} GB",
            option.label,
            option.description,
        )
    return table


def _format_saved_pct(pct: int) -> str:
    """Color the savings line without changing the underlying value."""
    style = "green" if pct > 0 else "yellow"
    return f"[{style}]{pct}%[/{style}]"


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


def _ensure_local_ollama_ready(cfg: Config, decision, force_local: bool) -> None:
    """Start Ollama and pull the selected local model when needed."""
    if decision.provider.name != "ollama":
        return
    if not (force_local or cfg.mode == "local"):
        return

    if not decision.provider.is_available():
        console.print("[yellow]Ollama is not running. Starting it in the background...[/yellow]")
        ensure_ollama_running(cfg.ollama_host)
        console.print("[green]Ollama is ready.[/green]")

    console.print(f"[yellow]Checking local model {decision.model}...[/yellow]")
    pulled = _ensure_ollama_model_with_progress(cfg.ollama_host, decision.model)
    if pulled:
        console.print(f"[green]Downloaded {decision.model}. The model is ready.[/green]")


def _ensure_ollama_model_with_progress(host: str, model: str) -> bool:
    """Pull an Ollama model with visible progress when it is missing."""
    task_id = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        def on_progress(event: dict) -> None:
            nonlocal task_id
            status = str(event.get("status") or "Downloading").capitalize()
            total = event.get("total")
            completed = event.get("completed")
            if total:
                if task_id is None:
                    task_id = progress.add_task(
                        f"{status} {model}",
                        total=int(total),
                    )
                progress.update(
                    task_id,
                    total=int(total),
                    completed=int(completed or 0),
                    description=f"{status} {model}",
                )
            elif task_id is None:
                task_id = progress.add_task(f"{status} {model}", total=None)
            else:
                progress.update(task_id, description=f"{status} {model}")

        pulled = ensure_ollama_model_available(
            host,
            model,
            progress_callback=on_progress,
        )

    return pulled


def _apply_decay_if_enabled(mem_store: MemoryStore, cfg: Config) -> None:
    """Apply memory decay lazily on every query when enabled."""
    if cfg.memory_decay_enabled:
        try:
            from .memory.decay import apply_decay
            apply_decay(mem_store)
        except Exception as e:
            from .utils.logging import get_logger
            get_logger(__name__).warning("memory decay failed", extra={"error": str(e)})


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
        if provider_name == "ollama":
            hint = (
                f"Ollama does not have the model '{model}' available locally.\n"
                f"  • Pull it:  ollama pull {model}\n"
                "  • Or set a known model:  memory-router config set local_model llama3.1:8b"
            )
        else:
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
    rows = [
        ("Provider", f"{provider_name} / {model}"),
        ("Route", reason),
        ("Memory", _memory_summary(used_memories)),
        ("Estimated saved", _format_saved_pct(est_saved_pct)),
    ]
    console.print(_summary_panel("Request summary", rows, border_style="cyan"))
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
        safe_msg = str(e).replace(str(Path.home()), "~")
        console.print(Panel(
            f"[red]Failed to build context:[/red] {safe_msg}\n\n"
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

    try:
        _ensure_local_ollama_ready(cfg, decision, local)
    except Exception as e:
        console.print(Panel(
            f"[red]Could not prepare local Ollama:[/red] {e}\n\n"
            "Install Ollama or check `memory-router doctor` for details.",
            title="Local model unavailable",
            border_style="red",
        ))
        raise typer.Exit(code=2)

    saved = percent_saved(built.full_history_tokens, built.sent_tokens)
    _print_routing_header(decision.provider.name, decision.model, decision.reason, built.used_memories, saved)
    if cfg.mode == "hybrid" and decision.provider.name != "ollama":
        console.print(Panel(
            "This prompt will be sent to a remote provider.",
            title="Hybrid routing note",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    # Stage 3: provider call with fallback.
    t0 = time.time()
    actual_provider = decision.provider.name

    if stream:
        result_text, real_in, real_out, actual_provider, actual_model = _stream_response(decision, built, router)
    else:
        try:
            if hasattr(router, "complete_with_fallback"):
                result, actual_provider, actual_model = router.complete_with_fallback(decision, built.messages)
            else:
                result = decision.provider.complete(decision.model, built.messages)
                actual_provider = decision.provider.name
                actual_model = decision.model
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
        console.print(Panel(result_text, title="Answer", border_style="green", box=box.ROUNDED, padding=(1, 2)))

    naive_in = built.full_history_tokens or built.sent_tokens
    real_saved = percent_saved(naive_in, real_in)
    cost = estimate_cost_usd(actual_model, real_in, real_out)

    token_rows = [
        ("Input tokens", f"{real_in:,}"),
        ("Output tokens", f"{real_out:,}"),
        ("Naive baseline", f"~{naive_in:,}"),
        ("Saved on input", _format_saved_pct(real_saved)),
        ("Latency", f"{latency_ms:,} ms"),
        ("Cost estimate", format_cost(cost)),
    ]
    if actual_provider != decision.provider.name:
        token_rows.append(("Fallback used", actual_provider))
    console.print(_summary_panel("Token usage", token_rows, border_style="green" if real_saved > 0 else "dim"))

    record_usage(
        kind="cli_ask",
        naive_tokens=naive_in,
        sent_tokens=real_in,
        output_tokens=real_out,
        memories_used=len(built.used_memories),
        provider=actual_provider,
        model=actual_model,
        cost_usd=cost,
    )

    # Record for adaptive routing
    _record_adaptive_outcome(cfg, router, decision, classification,
                             real_in, real_out, latency_ms, cost, None,
                             actual_provider=actual_provider,
                             actual_model=actual_model)

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
        # Fall back to non-streaming, including router-level provider fallback.
        if hasattr(router, "complete_with_fallback"):
            result, actual_provider, actual_model = router.complete_with_fallback(
                decision, built.messages
            )
        else:
            result = decision.provider.complete(decision.model, built.messages)
            actual_provider = decision.provider.name
            actual_model = decision.model
        text = result.text or "[no response]"
        console.print(text)
        return (
            text,
            result.input_tokens or built.sent_tokens,
            result.output_tokens or 0,
            actual_provider,
            actual_model,
        )

    console.print()  # newline after stream
    combined = "".join(full_text)
    if not real_in:
        real_in = built.sent_tokens
    return combined, real_in, real_out, decision.provider.name, decision.model


def _record_adaptive_outcome(
    cfg,
    router,
    decision,
    classification,
    input_tokens,
    output_tokens,
    latency_ms,
    cost,
    error,
    actual_provider: Optional[str] = None,
    actual_model: Optional[str] = None,
):
    """Record outcome for adaptive routing if enabled."""
    if not cfg.adaptive_routing:
        return
    try:
        from .adaptive_router import AdaptiveRouter, RouteOutcome
        if isinstance(router, AdaptiveRouter):
            # Estimate quality from auto-capture success
            quality = 0.7 if error is None else 0.1
            router.record_outcome(RouteOutcome(
                provider=actual_provider or decision.provider.name,
                model=actual_model or decision.model,
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
    except Exception as e:
        from .utils.logging import get_logger
        get_logger(__name__).warning("adaptive routing outcome recording failed", extra={"error": str(e)})


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
        safe_msg = str(e).replace(str(Path.home()), "~")
        console.print(Panel(
            f"[red]Failed to build context:[/red] {safe_msg}\n\n"
            "This usually means a corrupted SQLite file under ~/.memory-router/.\n"
            "Try: [bold]memory-router memory clear --yes[/bold] or remove the directory and re-init.",
            title="Context build error",
            border_style="red",
        ))
        raise typer.Exit(code=2)

    saved = percent_saved(built.full_history_tokens, built.sent_tokens)
    summary_rows = [
        ("Classification", f"{classification.task} / {classification.domain}"),
        ("Concepts", ", ".join(classification.concepts) if classification.concepts else "none"),
        ("Memory", _memory_summary(built.used_memories)),
        ("Tokens", f"sent ~{built.sent_tokens:,}  baseline ~{built.full_history_tokens:,}  saved {_format_saved_pct(saved)}"),
    ]
    console.print(_summary_panel("Context summary", summary_rows, border_style="cyan"))
    console.print()

    if show_messages:
        console.print("[bold green]Optimized messages:[/bold green]")
        for i, m in enumerate(built.messages, 1):
            console.print(_render_message_panel(m, i))
        return

    if built.used_memories:
        console.print(_memory_table(built.used_memories, title="Relevant memories"))
        console.print()
    else:
        console.print("[dim]Relevant memory used: (none)[/dim]")
        console.print()

    optimized_prompt = _render_flat_prompt(built.messages)
    console.print("[bold green]Optimized prompt:[/bold green]")
    console.print(optimized_prompt)


# ---------- init ----------

@app.command()
def init():
    """Interactive first-time setup."""
    ensure_dirs()
    console.print(Panel.fit(
        "[bold]Memory Router setup[/bold]\n"
        "Everything is stored locally under " f"{ROOT_DIR}",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print(_mode_guide_panel())

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
        specs = detect_system_specs()
        recommendation = recommend_ollama_model(specs)
        console.print(_system_specs_panel(specs, recommendation))
        console.print(_local_model_options_table())
        default_local_model = normalize_ollama_model_name(
            cfg.local_model, recommendation.model
        )
        chosen_model = Prompt.ask(
            "Choose Ollama model",
            default=default_local_model,
        )
        cfg.local_model = normalize_ollama_model_name(chosen_model, recommendation.model)
        if cfg.local_model != chosen_model.strip():
            console.print(
                f"[yellow]That did not look like a model id, so I set {cfg.local_model} instead.[/yellow]"
            )

    # v2 features
    cfg.mycelium_enabled = Confirm.ask("Enable mycelium memory network?", default=True)
    cfg.memory_decay_enabled = Confirm.ask("Enable memory decay (stale memories auto-fade)?", default=True)

    save_config(cfg)
    summary_rows = [
        ("Mode", cfg.mode),
        ("Memory Palace", "enabled" if cfg.memory_enabled else "disabled"),
        ("Auto capture", "enabled" if cfg.auto_capture_memories else "disabled"),
        ("Mycelium", "enabled" if cfg.mycelium_enabled else "disabled"),
        ("Decay", "enabled" if cfg.memory_decay_enabled else "disabled"),
    ]
    if mode in ("local", "hybrid"):
        summary_rows.append(("Ollama host", cfg.ollama_host))
        summary_rows.append(("Local model", cfg.local_model or "auto"))
    console.print(_summary_panel("Setup complete", summary_rows, border_style="green"))
    console.print(f"[green]Wrote config to {CONFIG_PATH}.[/green]")
    if mode == "local":
        console.print(
            "[dim]Local mode will auto-start Ollama in the background the first time you use `memory-router ask`.[/dim]"
        )
    console.print("Next: [bold]memory-router \"Explain bond convexity\"[/bold]")


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
    table.box = box.ROUNDED
    table.header_style = "bold cyan"
    table.title_style = "bold cyan"
    table.row_styles = ["none", "dim"]
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
    table.box = box.ROUNDED
    table.header_style = "bold cyan"
    table.title_style = "bold cyan"
    table.row_styles = ["none", "dim"]
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    if is_initialized():
        cfg = load_config()
    else:
        from .health import check_health

        report = check_health()
        table.add_row("config file", "[red]missing[/red]", "Run: memory-router init")
        for check in report.checks[1:]:
            status_style = {
                "ok": "[green]ok[/green]",
                "warn": "[yellow]warn[/yellow]",
                "error": "[red]error[/red]",
            }.get(check.status, check.status)
            table.add_row(check.name, status_style, check.detail)
        console.print(table)
        raise typer.Exit(code=1)

    overview_rows = [
        ("Mode", cfg.mode),
        ("Memory Palace", "on" if cfg.memory_enabled else "off"),
        ("Auto capture", "on" if cfg.auto_capture_memories else "off"),
        ("Mycelium", "on" if cfg.mycelium_enabled else "off"),
        ("Decay", "on" if cfg.memory_decay_enabled else "off"),
        ("Adaptive routing", "on" if cfg.adaptive_routing else "off"),
    ]
    if cfg.mode in ("local", "hybrid"):
        overview_rows.append(("Local model", cfg.local_model or "auto"))
    console.print(_summary_panel("Overview", overview_rows, border_style="cyan"))
    console.print()
    from .health import check_health
    report = check_health()
    for check in report.checks:
        status_style = {
            "ok": "[green]ok[/green]",
            "warn": "[yellow]warn[/yellow]",
            "error": "[red]error[/red]",
        }.get(check.status, check.status)
        table.add_row(check.name, status_style, check.detail)

    if cfg.mycelium_enabled:
        try:
            ms = MemoryStore()
            mycelium = _get_mycelium(ms, cfg)
            if mycelium:
                stats = mycelium.stats()
                table.add_row("mycelium_edges", "[green]ok[/green]", f"{stats['edge_count']} edges, avg weight {stats['avg_weight']}")
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
    summary_rows = [
        ("Cases", f"{len(report.cases):,}"),
        ("Avg raw tokens", f"{report.raw_tokens_avg:.0f}"),
        ("Avg baseline", f"{report.baseline_tokens_avg:.0f}"),
        ("Avg optimized", f"{report.optimized_tokens_avg:.0f}"),
        ("Avg saved", f"{report.raw_saved_pct_avg:.1f}%"),
    ]
    if report.baseline_score_avg is not None and report.optimized_score_avg is not None:
        summary_rows.extend([
            ("Avg score baseline", f"{report.baseline_score_avg:.2f}"),
            ("Avg score optimized", f"{report.optimized_score_avg:.2f}"),
            ("Avg delta", f"{report.quality_delta_avg:.2f}"),
        ])
    console.print(_summary_panel("Benchmark summary", summary_rows, border_style="cyan"))
    console.print()

    table = Table(title="Memory Router Benchmark", show_lines=False)
    table.box = box.ROUNDED
    table.header_style = "bold cyan"
    table.title_style = "bold cyan"
    table.row_styles = ["none", "dim"]
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
    console.print(Panel(headline, title="Memory Router: Cumulative Savings", border_style="green", box=box.ROUNDED))

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
    console.print(_memory_table(mems, title="Memories"))


@memory_app.command("palace")
def memory_palace():
    """Show memories grouped by domain → task."""
    _require_init()
    store = MemoryStore()
    nodes = build_palace(store)
    if not nodes:
        console.print("[yellow]Memory Palace is empty.[/yellow]")
        return
    tree = Tree("[bold cyan]Memory Palace[/bold cyan]", guide_style="dim")
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
    console.print(_memory_table(mems, title=f"Search results for {_truncate_text(query, 48)}"))


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
                  f"Avg confidence: {stats['avg_confidence']:.3f}[/dim]")


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
    rows = [
        ("Edges", f"{stats['edge_count']:,}"),
        ("Avg weight", f"{stats['avg_weight']:.3f}"),
        ("Max weight", f"{stats['max_weight']:.3f}"),
        ("Connected nodes", f"{stats['connected_nodes']:,}"),
    ]
    console.print(_summary_panel("Mycelium Network", rows, border_style="cyan"))


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
    console.print(f"\n{mode} - Consolidation results:")
    rows = [
        ("Clusters found", str(result.clusters_found)),
        ("Memories merged", str(result.memories_merged)),
        ("Memories remaining", str(result.memories_remaining)),
    ]
    console.print(_summary_panel("Memory Consolidation", rows, border_style="cyan"))

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

    console.print(_memory_table(similar, title=f"Similar memories (threshold={threshold})"))


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
    table.box = box.ROUNDED
    table.header_style = "bold cyan"
    table.row_styles = ["none", "dim"]
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
    sys.argv = [sys.argv[0], *_rewrite_argv(sys.argv[1:])]
    app()


if __name__ == "__main__":
    entry()
