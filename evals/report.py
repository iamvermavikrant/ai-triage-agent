"""Eval report formatter — Rich console output + JSON persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
REPORTS_DIR = Path(__file__).parent / "reports"


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    results = report["results"]

    console.rule("[bold blue]AI Triage Agent -- Eval Report[/bold blue]")

    # ── Summary panel ───────────────────────────────────────────────────
    summary_lines = [
        f"Fixtures run : {summary['total']}",
        "",
        "[bold]Custom LLM-as-Judge[/bold]",
        f"  Passed     : {summary['custom_passed']} / {summary['total']}",
        f"  Pass rate  : {summary['custom_pass_rate']*100:.1f}%",
        f"  Avg score  : {summary['custom_avg_score']} / 10",
        "",
        "[bold]DeepEval[/bold]",
        f"  Passed     : {summary['deepeval_passed']} / {summary['total']}",
        f"  Pass rate  : {summary['deepeval_pass_rate']*100:.1f}%",
    ]
    console.print(Panel("\n".join(summary_lines), title="Summary", border_style="blue"))

    # ── Per-fixture table: Custom judge ─────────────────────────────────
    console.print("\n[bold yellow]Custom LLM-as-Judge Results[/bold yellow]")
    t1 = Table(box=box.ROUNDED, show_lines=True)
    t1.add_column("Fixture", style="bold", width=30)
    t1.add_column("Failure Type", width=16)
    t1.add_column("Score", justify="center", width=7)
    t1.add_column("Pass", justify="center", width=6)
    t1.add_column("Priority", justify="center", width=9)
    t1.add_column("Time(s)", justify="right", width=8)
    t1.add_column("Critique", width=55)

    for r in results:
        j = r["judgment"]
        score = j["weighted_total"]
        passed = j["pass"]
        rca = r.get("rca_report", {})
        priority = rca.get("priority", "-")
        failure_type = r.get("log_analysis", {}).get("failure_type") or "-"
        color = "green" if passed else "red"
        t1.add_row(
            r["fixture_id"],
            failure_type,
            f"[{color}]{score}[/{color}]",
            "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            priority,
            str(r["elapsed_s"]),
            j.get("critique", ""),
        )
    console.print(t1)

    # ── Per-fixture table: DeepEval ─────────────────────────────────────
    console.print("\n[bold yellow]DeepEval Results[/bold yellow]")
    t2 = Table(box=box.ROUNDED, show_lines=True)
    t2.add_column("Fixture", style="bold", width=34)
    t2.add_column("GEval-Correct", justify="center", width=14)
    t2.add_column("GEval-Action", justify="center", width=13)
    t2.add_column("GEval-Focus", justify="center", width=12)
    t2.add_column("Hallucination", justify="center", width=14)
    t2.add_column("Relevancy", justify="center", width=10)
    t2.add_column("Overall", justify="center", width=9)

    for r in results:
        de = r.get("deepeval", {})
        geval = de.get("geval", {})
        hall = de.get("hallucination", {})
        rel = de.get("answer_relevancy", {})
        overall = de.get("deepeval_overall_pass", False)

        def _fmt(d: dict[str, Any]) -> Text:
            if not d:
                return Text("-", style="dim")
            score = d.get("score", 0)
            passed = d.get("passed", False)
            style = "green" if passed else "red"
            return Text(f"{score:.2f}", style=style)

        t2.add_row(
            r["fixture_id"],
            _fmt(geval.get("rca_correctness", {})),
            _fmt(geval.get("fix_actionability", {})),
            _fmt(geval.get("no_scope_creep", {})),
            _fmt(hall),
            _fmt(rel),
            Text("PASS", style="green") if overall else Text("FAIL", style="red"),
        )
    console.print(t2)

    # ── DeepEval metric explanation ─────────────────────────────────────
    console.print(
        "\n[dim]DeepEval metrics: GEval scores 0-1 (pass >= 0.7) | "
        "Hallucination score = hallucination rate (pass < 0.3) | "
        "Relevancy score 0-1 (pass >= 0.7)[/dim]"
    )


def save_report(report: dict[str, Any], output_dir: Path | None = None) -> Path:
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"eval_report_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[dim]Report saved to {path}[/dim]")
    return path
