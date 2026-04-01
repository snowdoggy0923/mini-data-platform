"""CLI interface for the data platform agent."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

app = typer.Typer(
    name="agent",
    help="A natural language CLI agent for your data warehouse.",
    no_args_is_help=True,
)
console = Console()


def _resolve_paths(db: str) -> tuple[str, str]:
    """Resolve database path and project directory."""
    db_path = str(Path(db).resolve())
    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)
    # project_dir is the parent of the warehouse dir, or cwd
    project_dir = str(Path(db_path).parent.parent)
    return db_path, project_dir


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[red]ANTHROPIC_API_KEY environment variable not set.[/red]\n\n"
                "Set it with:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...",
                title="Missing API Key",
            )
        )
        raise typer.Exit(1)


def _make_agent(db: str):
    """Create a DataAgent instance."""
    from agent.agent import DataAgent

    _check_api_key()
    db_path, project_dir = _resolve_paths(db)

    with Status("[bold blue]Scanning warehouse metadata...", console=console):
        agent = DataAgent(db_path=db_path, project_dir=project_dir)

    console.print("[dim]Metadata loaded. Ready.[/dim]\n")
    return agent


def _run_query(agent, question: str) -> None:
    """Run a single question through the agent with streaming output."""
    text_buffer: list[str] = []
    live = Live(console=console, refresh_per_second=12, vertical_overflow="visible")
    status: Status | None = None

    def on_text(delta: str) -> None:
        if status is not None:
            status.stop()
        text_buffer.append(delta)
        full = "".join(text_buffer)
        live.update(Markdown(full))

    def on_tool_start(name: str, tool_input: dict) -> None:
        nonlocal status
        labels = {
            "execute_sql": "Executing SQL query",
            "list_tables": "Listing tables",
            "describe_table": "Describing table",
            "get_metadata_context": "Reading metadata",
        }
        label = labels.get(name, name)
        detail = ""
        if name == "execute_sql":
            sql = tool_input.get("sql", "")
            # Show first 80 chars of SQL
            detail = f": {sql[:80]}{'...' if len(sql) > 80 else ''}"
        elif name == "describe_table":
            detail = f": {tool_input.get('schema', '')}.{tool_input.get('table', '')}"

        if live.is_started:
            live.stop()
        status = Status(f"[bold cyan]{label}{detail}[/bold cyan]", console=console)
        status.start()

    def on_tool_end(name: str, result: str) -> None:
        nonlocal status
        if status is not None:
            status.stop()
            status = None
        # Reset text buffer for next assistant text
        text_buffer.clear()
        live.start()

    live.start()
    try:
        agent.chat(
            question,
            on_text=on_text,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
    finally:
        if status is not None:
            status.stop()
        if live.is_started:
            live.stop()

    # Final render
    if text_buffer:
        console.print(Markdown("".join(text_buffer)))
    console.print()


@app.command()
def ask(
    question: str = typer.Argument(help="The question to ask about your data."),
    db: str = typer.Option("warehouse/data.duckdb", help="Path to DuckDB database file."),
) -> None:
    """Ask a single question about your data warehouse."""
    agent = _make_agent(db)
    _run_query(agent, question)


@app.command()
def chat(
    db: str = typer.Option("warehouse/data.duckdb", help="Path to DuckDB database file."),
) -> None:
    """Start an interactive chat session with the data agent."""
    agent = _make_agent(db)

    console.print(
        Panel(
            "[bold]Data Platform Agent[/bold]\n"
            "Ask questions about your data in natural language.\n"
            "Type [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] to leave.",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        _run_query(agent, question)


if __name__ == "__main__":
    app()
