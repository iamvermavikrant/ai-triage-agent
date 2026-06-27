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
        │
        ▼
  MCP Tool Server
  ├── fetch_test_logs  (GitHub Actions / local / mock)
  └── get_git_diff     (GitHub API / local git / mock)
        │
        ▼
  LangGraph Workflow
  ├── Log Analyzer    →  structured failure signals
  ├── Diff Analyzer   →  implicated files + relevance scores
  └── RCA Synthesizer →  P0–P3 report with fix + owner
        │
        ▼
  Eval Harness (5 fixtures × LLM-as-judge scoring)
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

## Running Evals

```bash
python -m evals.harness
```

Scores each fixture against ground truth using an LLM judge. Pass threshold: **7.0 / 10**. See [docs/eval_methodology.md](docs/eval_methodology.md).

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
│   ├── harness.py                # Eval runner
│   ├── judge.py                  # LLM-as-judge scorer
│   └── report.py                 # Rich console + JSON output
├── tests/
│   ├── test_mcp_tools.py
│   ├── test_agents.py
│   └── test_graph.py
├── scripts/
│   └── run_triage.py             # CLI entry point
├── pyproject.toml
└── requirements.txt
```

## Configuration

All prompts are managed in [`config/prompts.yaml`](config/prompts.yaml) with version tracking. To tune agent behavior, edit the relevant prompt and bump its `version` field — no code changes required.

| Env Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `TRIAGE_MODEL` | `claude-sonnet-4-6` | Model for pipeline agents |
| `JUDGE_MODEL` | `claude-opus-4-8` | Model for eval judge |
| `GITHUB_TOKEN` | — | Required for real CI logs |
| `GITHUB_REPO` | `owner/repo` | Target repository |

## License

MIT
