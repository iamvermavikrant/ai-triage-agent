"""Eval report formatter — Rich console output + JSON persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
REPORTS_DIR = Path(__file__).parent / "reports"


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    results = report["results"]

    console.rule("[bold blue]AI Triage Agent — Eval Report[/bold blue]")
    console.print(
        f"\n[bold]Total:[/bold] {summary['total']}  "
        f"[green]Passed:[/green] {summary['passed']}  "
        f"[red]Failed:[/red] {summary['failed']}  "
        f"[yellow]Pass Rate:[/yellow] {summary['pass_rate']*100:.1f}%  "
        f"[cyan]Avg Score:[/cyan] {summary['avg_score']}/10\n"
    )

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Fixture", style="bold", width=32)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Pass", justify="center", width=6)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("Time (s)", justify="right", width=9)
    table.add_column("Critique", width=50)

    for r in results:
        j = r["judgment"]
        score = j["weighted_total"]
        passed = j["pass"]
        priority = r.get("rca_report", {}).get("priority", "—")
        score_color = "green" if passed else "red"

        table.add_row(
            r["fixture_id"],
            f"[{score_color}]{score}[/{score_color}]",
            "[green]✓[/green]" if passed else "[red]✗[/red]",
            priority,
            str(r["elapsed_s"]),
            j.get("critique", "")[:80],
        )

    console.print(table)


def save_report(report: dict[str, Any], output_dir: Path | None = None) -> Path:
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"eval_report_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[dim]Report saved to {path}[/dim]")
    return path
