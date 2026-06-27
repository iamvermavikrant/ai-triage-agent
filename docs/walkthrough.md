# Interview Walkthrough Guide

This document explains the full project end-to-end in plain language —
what every term means, how to run it, and what to say during the interview.

---

## Table of Contents

1. [What this project does in one sentence](#1-what-this-project-does-in-one-sentence)
2. [Glossary — every term explained](#2-glossary--every-term-explained)
3. [How to run it step by step](#3-how-to-run-it-step-by-step)
4. [Full workflow walkthrough](#4-full-workflow-walkthrough)
5. [Understanding the eval report](#5-understanding-the-eval-report)
6. [The 5 test fixtures explained](#6-the-5-test-fixtures-explained)
7. [What the critique column means](#7-what-the-critique-column-means)
8. [How to explain this in 2 minutes](#8-how-to-explain-this-in-2-minutes)

---

## 1. What this project does in one sentence

> When a CI test fails, this agent automatically reads the error log and the
> code change that caused it, figures out the root cause, and writes a
> structured report telling you exactly what broke, why, and how to fix it.

---

## 2. Glossary — every term explained

### CI (Continuous Integration)
An automated system (like GitHub Actions) that runs your tests every time
someone pushes code. If tests fail, CI notifies the team.

### Test Log
The raw text output produced when a test runs — error messages, stack traces,
timing info. Example: "RuntimeError: CUDA out of memory at trainer.py line 312".

### Git Diff
A side-by-side comparison of what changed in the code between the previous
version and the current one. Shows exactly which lines were added or removed.

### RCA (Root Cause Analysis)
A structured report that answers: What broke? Why did it break? What is the
fix? Who owns it? In software, this is written after an incident or test failure.

### MCP (Model Context Protocol)
An open standard (like a USB-C port for AI) that lets AI models call external
tools in a structured way. We built an MCP server that exposes two tools:
`fetch_test_logs` and `get_git_diff`. The AI agents call these tools to gather
information before analysing anything.

### LangGraph
A Python library for building multi-step AI pipelines where each step is a
separate "agent" (node). It manages the flow of data between agents and handles
errors. Think of it like a flowchart that runs AI models at each decision point.

### Agent (in AI context)
A self-contained unit that receives input, calls an LLM (language model), and
produces structured output. This project has three agents:
- **Log Analyzer** — reads the error log
- **Diff Analyzer** — reads the code change
- **RCA Synthesizer** — combines both into a final report

### LLM (Large Language Model)
The AI brain — in this project that is Claude (made by Anthropic). It reads
text and produces text. We give it a detailed instruction (system prompt) and
the data to analyse (user prompt), and it returns a JSON response.

### System Prompt
The instruction we give the LLM before showing it the data. For example, the
Log Analyzer system prompt says: "You are an expert log analyst. Extract
failure signals and return them as JSON with these exact fields..."

### Prompt Management (config/prompts.yaml)
All system prompts are stored in a single YAML file with version numbers.
This means you can improve the AI's behaviour by editing one file — no code
changes needed. This is an enterprise pattern called "prompt versioning".

### Mock Mode (MOCK_LLM=true)
Instead of calling the real Claude API (which costs money and needs a key),
mock mode returns pre-written realistic responses. Used for demos and testing.
Set in the `.env` file. Everything still runs end-to-end — only the LLM call
is replaced.

### DeepEval
An open-source Python library for evaluating AI outputs. Provides standardised
metrics like GEval (custom criteria scoring), HallucinationMetric (does the AI
invent things?), and AnswerRelevancyMetric (is the answer on-topic?).

### LLM-as-Judge
Using a second (usually more powerful) LLM to score the output of the first
LLM. Like a senior engineer reviewing a junior engineer's report. In this
project, `claude-opus-4-8` reads the generated RCA and scores it on 5
dimensions against the known correct answer.

### Fixture
A pre-written test scenario with a known correct answer. We have 5 fixtures,
each simulating a different type of GPU test failure. Used to test whether the
agent produces the right RCA.

### GitHub Actions
A CI/CD system built into GitHub. When a test fails, our workflow
(`ci.yml`) automatically triggers the triage agent to analyse the failure and
upload the RCA report as an artifact.

---

## 3. How to run it step by step

### Prerequisites
- Python 3.10 or higher installed
- Git installed
- The repo already cloned at `C:\Users\Vikrant\Documents\ai-triage-agent`

### Step 1 — Install dependencies (one time only)
```bash
cd C:\Users\Vikrant\Documents\ai-triage-agent
pip install langgraph langchain-anthropic pyyaml rich python-dotenv structlog tenacity anthropic
pip install -e .
```

### Step 2 — Check the .env file
The `.env` file already exists with mock mode enabled:
```
MOCK_LLM=true
ANTHROPIC_API_KEY=sk-ant-dummy-key-not-needed-in-mock-mode
```
No changes needed to run in demo mode.

### Step 3 — Run the triage agent on a single failure
```bash
python scripts/run_triage.py \
  --run-id cuda_oom \
  --commit a3f1c2b9 \
  --branch feature/larger-batch \
  --test-suite test_model_training \
  --log-backend mock \
  --diff-backend mock
```
This simulates a real CI failure. You will see:
- The agent fetching logs and diff
- Each of the 3 agents running in sequence
- The final RCA report printed in a Rich panel

### Step 4 — Run the full eval suite (all 5 fixtures)
```bash
# Run both judges
python -m evals.harness --judge both

# Run only the custom LLM-as-judge
python -m evals.harness --judge custom

# Run only DeepEval metrics
python -m evals.harness --judge deepeval
```

### Step 5 — Run the unit tests
```bash
pytest tests/ -v
```

### Available fixture IDs (use with --run-id)
| run-id | What it simulates |
|--------|------------------|
| `cuda_oom` | GPU runs out of memory |
| `import_error` | A Python package import fails |
| `regression_diff` | Performance drops after a code change |
| `flaky_timeout` | A distributed test randomly times out |
| `env_mismatch` | Wrong CUDA version on the CI machine |

---

## 4. Full workflow walkthrough

Here is exactly what happens when you run `python -m evals.harness`:

```
Step 1: Load fixtures
  └── Read 5 JSON files from evals/fixtures/
      Each has: input (run_id, commit, branch) + ground_truth (correct RCA)

Step 2: For each fixture — fetch inputs via MCP tools
  ├── fetch_test_logs(run_id="cuda_oom", backend="mock")
  │     Returns: raw error log text (e.g. "CUDA out of memory at trainer.py:312")
  └── get_git_diff(commit_sha="a3f1c2b9", backend="mock")
        Returns: unified diff showing what lines changed

Step 3: Run the LangGraph triage pipeline
  │
  ├── Node 1: Log Analyzer
  │     Input:  raw log text
  │     Calls:  Claude (or mock) with log_analyzer system prompt
  │     Output: { failure_type: "CUDA_OOM", severity: "CRITICAL",
  │               stack_frames: [...], affected_modules: [...] }
  │
  ├── Node 2: Diff Analyzer
  │     Input:  log_analysis output + git diff
  │     Calls:  Claude (or mock) with diff_analyzer system prompt
  │     Output: { implicated_files: [...], change_risk: "HIGH",
  │               regression_likely: true, confidence: 0.95 }
  │
  └── Node 3: RCA Synthesizer
        Input:  log_analysis + diff_analysis + metadata
        Calls:  Claude (or mock) with rca_synthesizer system prompt
        Output: { title: "...", root_cause: "...", recommended_fix: "...",
                  priority: "P1", owner_hint: "ml-training team" }

Step 4: Score with Judge 1 — Custom LLM-as-Judge
  ├── Sends generated RCA + ground truth to claude-opus-4-8
  └── Returns: { accuracy: 9, actionability: 9, ... weighted_total: 9.1, pass: true }

Step 5: Score with Judge 2 — DeepEval
  ├── GEval (RCA Correctness, Fix Actionability, No Scope Creep)
  ├── HallucinationMetric
  └── AnswerRelevancyMetric

Step 6: Print report + save JSON to evals/reports/
```

### State passed between agents (TriageState)

Every agent reads from and writes to a shared state object:

```python
{
  # Inputs (set at the start)
  "run_id":       "fixture_01_cuda_oom",
  "raw_log":      "ERROR: CUDA out of memory...",
  "git_diff":     "- BATCH_SIZE = 8\n+ BATCH_SIZE = 64",
  "commit_sha":   "a3f1c2b9",
  "branch":       "feature/larger-batch",
  "test_suite":   "test_model_training",

  # Added by Log Analyzer
  "log_analysis": { "failure_type": "CUDA_OOM", ... },

  # Added by Diff Analyzer
  "diff_analysis": { "change_risk": "HIGH", ... },

  # Added by RCA Synthesizer
  "rca_report": { "priority": "P1", "recommended_fix": "..." },

  # Bookkeeping
  "errors":           [],
  "completed_nodes":  ["log_analyzer", "diff_analyzer", "rca_synthesizer"]
}
```

---

## 5. Understanding the eval report

When you run `python -m evals.harness --judge both`, you see two tables.

### Table 1 — Custom LLM-as-Judge Results

| Column | What it means |
|--------|--------------|
| Fixture | The name of the test scenario |
| Failure Type | What kind of error was detected (see section 6) |
| Score | Weighted average score out of 10 (pass = >= 7.0) |
| Pass | PASS if score >= 7.0, FAIL if below |
| Priority | P0/P1/P2/P3 — how urgent the issue is |
| Time(s) | How long the full pipeline took |
| Critique | The judge's written feedback on the RCA quality |

### Table 2 — DeepEval Results

| Column | What it means |
|--------|--------------|
| GEval-Correct | Did the agent identify the right root cause? (0-1, higher is better) |
| GEval-Action | Is the recommended fix specific enough to act on? (0-1) |
| GEval-Focus | Does the RCA stay on topic without adding irrelevant content? (0-1) |
| Hallucination | Rate of invented facts — file paths, functions, errors (0-1, LOWER is better) |
| Relevancy | Does the RCA actually address the test failure described? (0-1) |
| Overall | PASS if all 5 metrics meet their threshold, FAIL if any one fails |

### Priority levels explained

| Priority | Meaning | Example |
|----------|---------|---------|
| P0 | Everything is broken, all tests failing | Import error blocks test collection entirely |
| P1 | Critical feature broken, major tests failing | CUDA OOM blocks all training tests |
| P2 | One feature broken, intermittent failures | Flaky NCCL timeout (15% failure rate) |
| P3 | Minor issue, rarely fails | Cosmetic assertion or slow test |

---

## 6. The 5 test fixtures explained

### Fixture 1 — CUDA OOM (Out of Memory)
**What happened:** A developer changed `BATCH_SIZE` from 8 to 64 in `trainer.py`
(8 times larger). Each training batch now needs 14 GB of GPU memory, but the
GPU only has 12 GB free after loading the model weights. Python crashes with
"CUDA out of memory."

**CUDA:** CUDA is NVIDIA's parallel computing platform. GPU training in PyTorch
uses CUDA to run thousands of calculations simultaneously. When you ask for more
memory than the GPU has, you get a RuntimeError.

**Batch Size:** How many training examples are processed in one pass. Larger
batches need more GPU memory but can train faster. 8→64 is an 8x increase.

**Root cause:** BATCH_SIZE = 64 exceeds GPU VRAM (Video RAM) budget.
**Fix:** Revert to BATCH_SIZE = 8, or use gradient accumulation.
**Gradient Accumulation:** A technique to simulate a large batch size by splitting
it into smaller chunks and accumulating the gradients — same effect, less memory.

---

### Fixture 2 — Import Error
**What happened:** A developer renamed an import in `attention.py` from
`nvidia_flash_attention.FlashAttention` to `nvidia_attention.FlashAttentionKernel`.
The new package/symbol does not exist. Python cannot even load the test file,
so the entire test suite fails to collect (zero tests run).

**Import Error:** Python's way of saying "I can't find the thing you're trying
to import." Happens when a package is not installed or a class was renamed.

**Flash Attention:** A highly optimised algorithm for computing attention in
transformer models (like GPT). It runs much faster on NVIDIA GPUs.

**Root cause:** Wrong import symbol name.
**Fix:** Revert the import line to the original package name.

---

### Fixture 3 — Throughput Regression
**What happened:** Two changes were made in one PR:
1. The attention calculation was rewritten using `torch.einsum` but the
   `1/sqrt(d_k)` scaling factor was accidentally left out.
2. The learning rate was multiplied by 10x.

Both cause the model to perform badly, resulting in 38.7% throughput drop
(1420 → 871 tokens/sec).

**Throughput:** How fast the model processes text, measured in tokens per second.
A 38% drop is severe — it means the same job takes 60% longer.

**Attention Score Scaling (1/sqrt(d_k)):** In transformer models, attention scores
are divided by the square root of the key dimension to prevent them from becoming
too large. Without it, the softmax function gets extremely sharp values (near 0 or 1),
which breaks learning.

**Learning Rate:** Controls how big each update step is during training. A 10x
increase causes the model to overshoot optimal values and diverge.

**Root cause:** Missing scale factor + aggressive learning rate.
**Fix:** Add `/ math.sqrt(d_k)` back to the einsum, revert LR multiplier.

---

### Fixture 4 — Flaky NCCL Timeout
**What happened:** During distributed training across 4 GPUs, one GPU (rank 2)
stopped responding during a synchronisation step called `allreduce`. After
waiting 1800 seconds (30 minutes), the process gave up and failed. This has
happened 3 times in the last 20 test runs (15% failure rate).

**NCCL (NVIDIA Collective Communications Library):** A library that coordinates
multiple GPUs when training a model across several cards or machines. Every GPU
must complete each step before any can move to the next.

**Distributed Training:** Training a model on multiple GPUs simultaneously by
splitting the work. Requires all GPUs to stay in sync using "collective operations"
like allreduce.

**Allreduce:** A collective operation where each GPU shares its gradients with
all other GPUs, and all receive the sum. If one GPU is slow or offline, everyone waits.

**Flaky test:** A test that sometimes passes and sometimes fails without any code
change. Often caused by hardware issues, network problems, or race conditions.

**Root cause:** Likely GPU thermal throttling or NVLink issue on rank 2 node.
**Why this gets a FAIL score (6.7):** The root cause is speculative (we can't
confirm it without hardware investigation). The fix is too vague ("add retry logic").
This is realistic — flaky distributed failures are genuinely hard to diagnose.

---

### Fixture 5 — Environment Mismatch
**What happened:** A developer added BF16 (Brain Float 16) support, which
requires CUDA version 12.1 or higher. They updated `cuda_check.py` to assert
this minimum version. But the CI runner's Docker image still has CUDA 11.8.
The check fails immediately, before any tests even start.

**CUDA Version:** NVIDIA releases new versions of CUDA with new features.
CUDA 12.1 added better support for BF16 tensor cores (special hardware for
fast AI calculations).

**BF16 (Brain Float 16):** A number format used in AI training. Uses less
memory than full 32-bit floats but keeps a wide range of values. Requires
specific GPU hardware support (CUDA 12.1+).

**Docker Image:** A pre-packaged environment containing the OS, libraries,
and tools needed to run the code. CI uses Docker to ensure a consistent
environment. If the Docker image has CUDA 11.8 but the code needs 12.1,
everything fails.

**Root cause:** CI Docker image not updated to match new CUDA requirement.
**Fix:** Update Docker image to `nvidia/cuda:12.1.0-devel-ubuntu22.04`,
or make the BF16 path optional using `torch.cuda.is_bf16_supported()`.

---

## 7. What the critique column means

The critique is written by the LLM judge after comparing the generated RCA
against the known correct answer. Here is what each fixture's critique means:

### fixture_01_cuda_oom — Score 9.1 PASS
> "Root cause precisely identifies the 8x batch size increase as the OOM trigger.
> Fix is immediately actionable. Preventive measures are enforceable in CI."

**What this means:** The agent correctly said "BATCH_SIZE went from 8 to 64,
that is why the GPU ran out of memory." The fix (gradient accumulation) is
specific enough for an engineer to implement immediately. Score is high because
all five dimensions are strong.

---

### fixture_02_import_error — Score 9.6 PASS (highest)
> "Perfect identification of the renamed import symbol as root cause.
> One-line fix is crystal clear. Minor: does not suggest adding an import smoke test."

**What this means:** The agent nailed it — wrong import, here's the line to fix.
Loses a tiny amount on completeness because it didn't suggest a preventive measure
(adding a test that imports all modules as a CI smoke test).

---

### fixture_03_regression_diff — Score 8.2 PASS
> "Correctly flags missing scale factor and LR multiplier. Fix is actionable
> but could specify the exact line to change. Blast radius slightly conservative."

**What this means:** The agent found both problems (scale factor + LR). The fix
is correct but slightly vague ("restore scale factor" could be more specific:
"add `/ math.sqrt(d_k)` on line 44 of scaled_dot_product.py"). Blast radius
(what else is affected) is understated.

---

### fixture_04_flaky_timeout — Score 6.7 FAIL
> "Intermittent nature correctly identified. Root cause is speculative — thermal
> throttling not confirmed. Fix lacks specificity: 'add retry logic' is too vague
> for a P2 incident."

**What this means:** The agent correctly noticed it is a flaky test (not
reproducible). But it guessed "thermal throttling" without evidence, and the
fix is not specific enough. This **intentional FAIL** shows the judge is honest —
it does not just give everything a high score. Flaky distributed failures are
genuinely the hardest type to diagnose.

---

### fixture_05_env_mismatch — Score 9.1 PASS
> "Correctly attributes failure to CUDA 11.8 vs 12.1 mismatch. Both fix paths
> are actionable. Excellent preventive measure suggestion."

**What this means:** The agent correctly identified the version gap and offered
two valid fixes (update Docker image, or use a runtime capability check).
The preventive measure (add environment version matrix to CI) is something a
senior DevOps engineer would suggest.

---

## 8. How to explain this in 2 minutes

Use this script during the interview:

---

*"The AI Triage Agent automates root cause analysis for GPU test failures.*

*When a CI test fails, the agent first fetches the error log and git diff using
two MCP tools I built — fetch_test_logs and get_git_diff. These follow the
Model Context Protocol, which is an open standard for giving AI models access
to external data.*

*The data flows through a three-node LangGraph pipeline. The Log Analyzer
reads the error log and classifies the failure — CUDA OOM, import error,
timeout, and so on. The Diff Analyzer correlates the code changes with the
failure signals. The RCA Synthesizer combines both into a structured report
with a priority level, root cause, recommended fix, and owner hint.*

*All prompts are managed in a versioned YAML file so we can tune the agent
without touching Python code.*

*For evaluation I built a dual harness. The first judge is a custom
LLM-as-judge powered by Claude Opus — it scores each RCA on five weighted
dimensions with a pass threshold of 7.0 out of 10. The second is DeepEval,
an industry-standard framework, which adds hallucination detection and answer
relevancy scoring.*

*I have five fixtures covering the most common GPU failure types we see at
NVIDIA — CUDA OOM, import errors, throughput regressions, flaky NCCL timeouts,
and environment mismatches. Four out of five pass both judges. The timeout
fixture intentionally fails — flaky distributed failures are genuinely hard
to diagnose automatically, and I wanted the evaluator to reflect that honestly.*

*Everything runs in mock mode without an API key for the demo, and switches to
real Claude calls with a single environment variable."*

---
