"""memory-router CLI.

This is the user-facing surface. Everything else is library code and could be
embedded in another app. Built with Typer for clean subcommands and Rich for
readable output.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from . import __version__
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
from .memory.palace import build_palace
from .memory.sqlite_store import ConversationStore, Memory, MemoryStore, Message
from .router import Router
from .security.keychain import delete_secret, get_secret, set_secret
from .utils.tokens import percent_saved


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


def _print_routing_header(model: str, used_memories, est_saved_pct: int):
    if used_memories:
        m = used_memories[0]
        memory_path = f"{m.task.title()} > {m.domain.title()}"
        if m.concepts:
            memory_path += " > " + ", ".join(m.concepts[:2])
    else:
        memory_path = "(none)"
    console.print(f"[bold]Using:[/bold] {model}")
    console.print(f"[bold]Memory used:[/bold] {memory_path}")
    console.print(f"[bold]Estimated tokens saved:[/bold] {est_saved_pct}%")
    console.print()


# ---------- top-level commands ----------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Ask a question directly."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip memory retrieval for this query."),
    local: bool = typer.Option(False, "--local", help="Force local model only."),
    session: str = typer.Option("default", "--session", help="Conversation session id."),
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
):
    """Default action: if a query is given, ask it. Otherwise show help."""
    if version:
        console.print(f"memory-router {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    if not query:
        console.print(ctx.get_help())
        raise typer.Exit()

    _ask(query=query, no_memory=no_memory, local=local, session=session)


def _ask(query: str, no_memory: bool, local: bool, session: str):
    cfg = _require_init()
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

    router = Router(cfg)
    decision = router.route(classification, force_local=local)

    saved = percent_saved(built.full_history_tokens, built.sent_tokens)
    _print_routing_header(decision.model, built.used_memories, saved)

    try:
        result = decision.provider.complete(decision.model, built.messages)
    except Exception as e:
        console.print(f"[red]Provider error:[/red] {e}")
        raise typer.Exit(code=2)

    console.print(Panel(result.text or "[no response]", title="Answer", border_style="green"))

    # Log the turn locally so future queries benefit from short-term memory.
    conv_store.add(Message(session_id=session, role="user", content=query))
    conv_store.add(Message(session_id=session, role="assistant", content=result.text or ""))


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

    # Flat copy-paste prompt: concatenate system notes, then the user query.
    system_blocks = [m["content"] for m in built.messages if m.get("role") == "system"]
    user_blocks = [m["content"] for m in built.messages if m.get("role") == "user"]
    parts = []
    if system_blocks:
        parts.append("\n\n".join(system_blocks))
    parts.append(user_blocks[-1] if user_blocks else query)
    optimized_prompt = "\n\n---\n\n".join(parts)

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

    if mode in ("api", "hybrid"):
        if Confirm.ask("Add an OpenAI API key now?", default=False):
            key = Prompt.ask("OpenAI API key", password=True)
            backend = set_secret("openai", key)
            console.print(f"[green]Saved OpenAI key to {backend}.[/green]")
        if Confirm.ask("Add an Anthropic API key now?", default=False):
            key = Prompt.ask("Anthropic API key", password=True)
            backend = set_secret("anthropic", key)
            console.print(f"[green]Saved Anthropic key to {backend}.[/green]")

    if mode in ("local", "hybrid"):
        host = Prompt.ask("Ollama host", default=cfg.ollama_host)
        cfg.ollama_host = host

    save_config(cfg)
    console.print(f"[green]Wrote config to {CONFIG_PATH}.[/green]")
    console.print("Try: [bold]memory-router \"Explain bond convexity\"[/bold]")


# ---------- auth ----------

@app.command()
def auth(
    provider: str = typer.Argument(..., help="Provider name: openai | anthropic | ..."),
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


if __name__ == "__main__":
    app()
