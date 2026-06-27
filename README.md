# AI Triage Agent

> Enterprise-grade automated root-cause analysis for CI test failures in GPU-accelerated ML workloads.

[![CI](https://github.com/iamvermavikrant/ai-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/iamvermavikrant/ai-triage-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

When a test suite fails in CI, the AI Triage Agent:

1. **Fetches** the raw test logs and git diff via MCP tools
2. **Analyzes** the logs to extract structured failure signals (CUDA OOM, import errors, regressions, timeouts, env mismatches)
3. **Correlates** the diff to identify which code changes caused the failure
4. **Synthesizes** a prioritized RCA report with root cause, recommended fix, blast radius, and owner hints
5. **Scores** the report quality using an LLM-as-judge eval harness

## Architecture

```
GitHub Actions failure
        |
        v
  MCP Tool Server
  |-- fetch_test_logs  (GitHub Actions / local / mock)
  +-- get_git_diff     (GitHub API / local git / mock)
        |
        v
  LangGraph Workflow
  |-- Log Analyzer    -->  structured failure signals
  |-- Diff Analyzer   -->  implicated files + relevance scores
  +-- RCA Synthesizer -->  P0-P3 report with fix + owner
        |
        v
  Dual Eval Harness
  |-- Custom LLM-as-Judge  (5 weighted dimensions, Claude Opus)
  +-- DeepEval             (GEval x3 + Hallucination + Relevancy)
```

See [docs/architecture.md](docs/architecture.md) for the full design.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | [Anthropic Claude](https://www.anthropic.com/) (`claude-sonnet-4-6` for agents, `claude-opus-4-8` for judge) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-agent graph |
| Tool Protocol | [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) — stdio server |
| LLM SDK | [anthropic-python](https://github.com/anthropics/anthropic-sdk-python) + [langchain-anthropic](https://github.com/langchain-ai/langchain) |
| Prompt Management | YAML-based versioned prompt library (`config/prompts.yaml`) |
| API / Server | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev/) |
| CLI Output | [Rich](https://github.com/Textualize/rich) — styled terminal panels and tables |
| Logging | [structlog](https://www.structlog.org/) — structured key-value logging |
| Retry Logic | [Tenacity](https://tenacity.readthedocs.io/) — exponential backoff on LLM calls |
| HTTP Client | [HTTPX](https://www.python-httpx.org/) — async-ready GitHub API calls |
| Git Integration | [GitPython](https://gitpython.readthedocs.io/) — local diff extraction |
| Config | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Eval Framework | [DeepEval](https://docs.confident-ai.com/) — GEval, HallucinationMetric, AnswerRelevancyMetric |
| Testing | [pytest](https://pytest.org/) + [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) + [pytest-cov](https://pytest-cov.readthedocs.io/) |
| Linting / Types | [Ruff](https://docs.astral.sh/ruff/) + [mypy](https://mypy.readthedocs.io/) |
| CI/CD | GitHub Actions |
| Language | Python 3.10+ |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/iamvermavikrant/ai-triage-agent.git
cd ai-triage-agent
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

# 3. Run against a mock fixture
python scripts/run_triage.py \
  --run-id cuda_oom \
  --commit a3f1c2b9 \
  --branch feature/larger-batch \
  --test-suite test_model_training \
  --log-backend mock \
  --diff-backend mock

# 4. Run against a real GitHub Actions failure
python scripts/run_triage.py \
  --run-id 12345678 \
  --commit abc123def456 \
  --branch main \
  --test-suite test_model_training
```

## Running Tests

```bash
pytest tests/ -v
```

## Evaluation

The project uses a **dual eval harness** — both judges run on every fixture automatically.

### Judge 1 — Custom LLM-as-Judge (`evals/judge.py`)

Powered by `claude-opus-4-8`. Scores the RCA on five weighted dimensions:

| Dimension | Weight | What it checks |
|-----------|--------|----------------|
| Accuracy | 35% | Root cause matches ground truth |
| Actionability | 25% | Fix is specific and executable |
| Completeness | 20% | All failure signals are addressed |
| Precision | 10% | No hallucinated files or functions |
| Clarity | 10% | Understandable to a developer |

**Pass threshold: 7.0 / 10** (weighted average)

### Judge 2 — DeepEval (`evals/deepeval_metrics.py`)

Industry-standard eval framework with three metric groups:

| Metric | Type | What it checks | Pass threshold |
|--------|------|----------------|----------------|
| `GEval – RCA Correctness` | GEval | Root cause matches the expected failure | score >= 0.7 |
| `GEval – Fix Actionability` | GEval | Recommended fix is specific, not vague | score >= 0.7 |
| `GEval – No Scope Creep` | GEval | RCA stays focused on the actual failure | score >= 0.7 |
| `HallucinationMetric` | Hallucination | No invented file paths or function names | rate < 0.3 |
| `AnswerRelevancyMetric` | Relevancy | RCA directly addresses the test failure | score >= 0.7 |

**GEval** uses LLM-graded criteria — each criterion is phrased as a natural-language rubric evaluated by the judge model, making it more flexible than exact-match scoring.

**HallucinationMetric** is especially important for RCA: a hallucinated file name or function sends engineers on a wild-goose chase.

### Running the full eval suite

```bash
# Mock mode (no API key needed)
python -m evals.harness

# Real mode (requires ANTHROPIC_API_KEY, set MOCK_LLM=false in .env)
MOCK_LLM=false python -m evals.harness
```

Results are printed as two Rich tables (one per judge) and saved to `evals/reports/eval_report_{timestamp}.json`.

See [docs/eval_methodology.md](docs/eval_methodology.md) for fixture details and scoring rationale.

## MCP Server

```bash
# Start in stdio mode (for Claude Desktop / MCP clients)
python -m ai_triage_agent.mcp.server
```

See [docs/mcp_setup.md](docs/mcp_setup.md) for Claude Desktop configuration.

## Project Structure

```
ai-triage-agent/
├── .github/workflows/
│   └── ci.yml                    # CI + auto-triage on failure
├── config/
│   └── prompts.yaml              # Versioned system/user prompts
├── docs/
│   ├── architecture.md
│   ├── mcp_setup.md
│   └── eval_methodology.md
├── src/ai_triage_agent/
│   ├── agents/
│   │   ├── log_analyzer.py       # LangGraph node 1
│   │   ├── diff_analyzer.py      # LangGraph node 2
│   │   └── rca_synthesizer.py    # LangGraph node 3
│   ├── graph/
│   │   ├── state.py              # TriageState TypedDict
│   │   └── workflow.py           # Compiled LangGraph
│   ├── mcp/
│   │   ├── server.py             # MCP server (stdio)
│   │   └── tools/
│   │       ├── fetch_test_logs.py
│   │       └── get_git_diff.py
│   └── utils/
│       ├── llm_client.py         # Anthropic SDK wrapper + retry
│       └── prompt_loader.py      # YAML prompt loader
├── evals/
│   ├── fixtures/                 # 5 JSON test scenarios
│   ├── harness.py                # Dual eval runner (both judges)
│   ├── judge.py                  # Custom LLM-as-judge (5 weighted dims)
│   ├── deepeval_metrics.py       # DeepEval: GEval + Hallucination + Relevancy
│   └── report.py                 # Rich console tables + JSON persistence
├── tests/
│   ├── test_mcp_tools.py
│   ├── test_agents.py
│   ├── test_graph.py
│   └── test_deepeval.py          # DeepEval unit + integration tests
├── scripts/
│   └── run_triage.py             # CLI entry point
├── pyproject.toml
└── requirements.txt
```

## Configuration

All prompts are managed in [`config/prompts.yaml`](config/prompts.yaml) with version tracking. To tune agent behavior, edit the relevant prompt and bump its `version` field — no code changes required.

| Env Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for real LLM calls |
| `MOCK_LLM` | `false` | Set `true` to run entire pipeline without any API key |
| `TRIAGE_MODEL` | `claude-sonnet-4-6` | Model for pipeline agents (Log/Diff/RCA) |
| `JUDGE_MODEL` | `claude-opus-4-8` | Model for custom LLM-as-judge eval |
| `GITHUB_TOKEN` | — | Required for real CI log + diff fetching |
| `GITHUB_REPO` | `owner/repo` | Target repository in `owner/repo` format |

## License

MIT
