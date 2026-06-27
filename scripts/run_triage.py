"""CLI entry point for running the AI Triage Agent on a specific CI failure."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

load_dotenv()

# Add src to path for editable-install-free runs
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_triage_agent.graph.workflow import triage_graph
from ai_triage_agent.mcp.tools.fetch_test_logs import fetch_test_logs
from ai_triage_agent.mcp.tools.get_git_diff import get_git_diff

console = Console()
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Triage Agent — CLI runner")
    p.add_argument("--run-id", required=True, help="CI run ID or fixture key")
    p.add_argument("--commit", required=True, help="Commit SHA")
    p.add_argument("--branch", default="unknown", help="Branch name")
    p.add_argument("--test-suite", default="unknown", help="Test suite name")
    p.add_argument("--log-backend", default="github_actions", choices=["github_actions", "local_file", "mock"])
    p.add_argument("--diff-backend", default="github_api", choices=["github_api", "local", "mock"])
    p.add_argument("--output", default=None, help="Path to write JSON report")
    return p.parse_args()


def main() -> None:
    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    args = parse_args()

    console.rule("[bold blue]AI Triage Agent[/bold blue]")
    console.print(f"[dim]Run ID:[/dim] {args.run_id}")
    console.print(f"[dim]Commit:[/dim] {args.commit}")
    console.print(f"[dim]Branch:[/dim] {args.branch}\n")

    # ── Fetch inputs ──────────────────────────────────────────────────────
    console.print("[yellow]► Fetching test logs...[/yellow]")
    raw_log = fetch_test_logs(run_id=args.run_id, backend=args.log_backend)

    console.print("[yellow]► Fetching git diff...[/yellow]")
    git_diff = get_git_diff(commit_sha=args.commit, backend=args.diff_backend)

    # ── Run graph ─────────────────────────────────────────────────────────
    console.print("[yellow]► Running triage pipeline...[/yellow]\n")
    final_state = triage_graph.invoke(
        {
            "run_id": args.run_id,
            "test_suite": args.test_suite,
            "branch": args.branch,
            "commit_sha": args.commit,
            "raw_log": raw_log,
            "git_diff": git_diff,
            "errors": [],
            "completed_nodes": [],
        }
    )

    errors = final_state.get("errors", [])
    rca = final_state.get("rca_report", {})

    if errors:
        console.print(f"[red]Pipeline errors:[/red] {errors}")
        sys.exit(1)

    # ── Display ───────────────────────────────────────────────────────────
    console.print(
        Panel(
            JSON(json.dumps(rca, indent=2)),
            title=f"[bold green]RCA Report — {rca.get('priority', '?')} — {rca.get('title', '')}[/bold green]",
            border_style="green",
        )
    )

    if args.output:
        Path(args.output).write_text(json.dumps(final_state, indent=2))
        console.print(f"\n[dim]Full state written to {args.output}[/dim]")

    # Completed nodes summary
    console.print(f"\n[dim]Completed nodes: {final_state.get('completed_nodes')}[/dim]")


if __name__ == "__main__":
    main()
