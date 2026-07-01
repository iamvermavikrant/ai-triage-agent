# Architecture

## Overview

The AI Triage Agent is a multi-agent system that automatically diagnoses CI test failures and produces structured Root Cause Analysis (RCA) reports. It combines an MCP tool server, a LangGraph orchestration layer, and an LLM-as-judge eval harness.

```
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions CI                          │
│   test failure → triage_on_failure job → run_triage.py CLI     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Tool Server                             │
│  ┌─────────────────────┐   ┌──────────────────────────────┐    │
│  │  fetch_test_logs    │   │       get_git_diff           │    │
│  │  (GitHub Actions /  │   │  (GitHub API / local git /   │    │
│  │   local / mock)     │   │   mock)                      │    │
│  └─────────────────────┘   └──────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ raw_log + git_diff
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                LangGraph Triage Workflow                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Log Analyzer │───▶│Diff Analyzer │───▶│ RCA Synthesizer  │  │
│  │              │    │              │    │                  │  │
│  │ Extracts     │    │ Correlates   │    │ Produces human-  │  │
│  │ structured   │    │ diff hunks   │    │ readable RCA     │  │
│  │ failure      │    │ with failure │    │ with priority,   │  │
│  │ signals      │    │ signals      │    │ fix, and owner   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  Shared State (TriageState TypedDict):                          │
│    raw_log, git_diff → log_analysis → diff_analysis → rca_report│
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Eval Harness                                  │
│  5 fixtures → pipeline → LLM-as-judge scoring → JSON report     │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Roles

### Log Analyzer
- **Input**: Raw CI/test log text
- **Output**: `LogAnalysis` — structured failure type, stack frames, severity, affected modules
- **Model**: `claude-sonnet-4-6` (fast, accurate for structured extraction)
- **Prompt key**: `log_analyzer` in `config/prompts.yaml`

### Diff Analyzer
- **Input**: `LogAnalysis` + unified git diff
- **Output**: `DiffAnalysis` — implicated files with relevance scores, change risk, confidence
- **Model**: `claude-sonnet-4-6`
- **Prompt key**: `diff_analyzer`
- **Edge case**: If no diff is available, returns a low-risk placeholder without calling the LLM.

### RCA Synthesizer
- **Input**: `LogAnalysis` + `DiffAnalysis` + metadata (branch, commit, test suite)
- **Output**: `RCAReport` — human-readable title, root cause, recommended fix, priority, owner hint
- **Model**: `claude-sonnet-4-6` with `max_tokens=4096` for detailed reports
- **Prompt key**: `rca_synthesizer`

## State Machine

LangGraph compiles the three nodes into a directed graph with conditional edges:

```
log_analyzer → [errors?] → END
                        ↓
              diff_analyzer → [errors?] → END
                                      ↓
                           rca_synthesizer → END
```

All state is passed via `TriageState` (a `TypedDict`), enabling stateless nodes and easy testing.

## Data Flow

```
TriageState {
  # Inputs
  run_id, test_suite, branch, commit_sha
  raw_log           ← fetch_test_logs MCP tool
  git_diff          ← get_git_diff MCP tool

  # Agent outputs (added sequentially)
  log_analysis      ← LogAnalyzerNode
  diff_analysis     ← DiffAnalyzerNode
  rca_report        ← RCASynthesizerNode

  # Bookkeeping
  errors[]
  completed_nodes[]
}
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph over raw chains | Built-in state management, conditional edges, easy visualization |
| TypedDict state | Zero runtime overhead, full mypy type safety, no Pydantic import cost |
| Prompts in YAML | Version-controlled, tunable without touching Python code |
| MCP for tool layer | Standard protocol; swappable backends (GitHub/local/mock) |
| LLM-as-judge | No hand-crafted rubrics; judge prompt is also versioned in YAML |
| `claude-opus-4-8` for judge | Highest accuracy for evaluation; `claude-sonnet-4-6` for production speed |
