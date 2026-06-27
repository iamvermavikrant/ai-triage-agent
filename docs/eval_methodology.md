# Eval Methodology

## Overview

The eval harness validates that the AI Triage Agent produces accurate, actionable RCA reports across a diverse set of GPU/ML failure scenarios. It uses **LLM-as-judge** scoring rather than exact-match string comparison, which is more robust for natural-language outputs.

## Fixtures

Five fixtures cover the most common failure categories in GPU-accelerated ML workloads:

| ID | Failure Type | Scenario |
|----|-------------|----------|
| `fixture_01_cuda_oom` | `CUDA_OOM` | Batch size increase causes GPU OOM |
| `fixture_02_import_error` | `IMPORT_ERROR` | Package rename breaks test collection |
| `fixture_03_regression_diff` | `ASSERTION_FAILED` | Missing attention scale factor causes 38.7% throughput drop |
| `fixture_04_flaky_timeout` | `TIMEOUT` | NCCL barrier timeout on rank 2 (intermittent) |
| `fixture_05_env_mismatch` | `ENV_MISMATCH` | CUDA 12.1 required but CI has 11.8 |

### Fixture Schema

```json
{
  "id": "fixture_01_cuda_oom",
  "description": "...",
  "input": {
    "run_id": "cuda_oom",
    "test_suite": "test_model_training",
    "branch": "feature/larger-batch",
    "commit_sha": "a3f1c2b9",
    "backend_log": "mock",
    "backend_diff": "mock"
  },
  "ground_truth": {
    "failure_type": "CUDA_OOM",
    "root_cause": "...",
    "priority": "P1",
    "recommended_fix": "...",
    "implicated_file": "src/training/trainer.py"
  }
}
```

## LLM-as-Judge Scoring

The judge model (`claude-opus-4-8`) evaluates the generated RCA against the ground truth on five dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `accuracy` | 35% | Does the root cause match ground truth? |
| `completeness` | 20% | Are all failure signals addressed? |
| `actionability` | 25% | Is the fix specific and executable? |
| `precision` | 10% | No hallucinated files/functions/errors? |
| `clarity` | 10% | Understandable to a developer? |

**Weighted total** = Σ(score × weight), scaled 0–10.

**Pass threshold**: `weighted_total >= 7.0`

### Rationale for weights

- **Accuracy** is weighted highest because an incorrect root cause leads to wasted engineering time.
- **Actionability** ranks second — a vague fix is nearly as bad as a wrong one.
- **Completeness** ensures the agent doesn't fixate on one signal and miss another.
- **Precision** prevents hallucinations from eroding trust in the tool.
- **Clarity** is weighted lowest because developers can parse technical output even if it's terse.

## Running Evals

```bash
# Full eval suite (requires ANTHROPIC_API_KEY)
python -m evals.harness

# Single fixture
python -c "
from evals.harness import load_fixtures, run_fixture
from evals.report import print_report
fixtures = [f for f in load_fixtures() if f['id'] == 'fixture_01_cuda_oom']
result = run_fixture(fixtures[0])
print_report({'summary': {}, 'results': [result]})
"
```

## Interpreting Results

The Rich-formatted report shows:

```
╭──────────────────────────────────┬───────┬──────┬──────────┬──────────┬──────────────╮
│ Fixture                          │ Score │ Pass │ Priority │ Time (s) │ Critique     │
├──────────────────────────────────┼───────┼──────┼──────────┼──────────┼──────────────┤
│ fixture_01_cuda_oom              │ 8.7   │  ✓   │ P1       │ 12.4     │ Accurate...  │
│ fixture_02_import_error          │ 9.1   │  ✓   │ P0       │ 9.8      │ Precise...   │
│ fixture_03_regression_diff       │ 7.8   │  ✓   │ P1       │ 14.2     │ Good...      │
│ fixture_04_flaky_timeout         │ 6.9   │  ✗   │ P2       │ 11.1     │ Fix too vag. │
│ fixture_05_env_mismatch          │ 8.4   │  ✓   │ P1       │ 10.3     │ Strong...    │
╰──────────────────────────────────┴───────┴──────┴──────────┴──────────┴──────────────╯
```

Reports are persisted as JSON in `evals/reports/eval_report_{timestamp}.json` for historical comparison.

## CI Integration

The eval harness runs automatically when the `test` job fails (via the `triage-on-failure` GitHub Actions job). The report is uploaded as an artifact named `triage-report-{run_id}` and retained for 30 days.

## Extending Fixtures

To add a new fixture:

1. Create `evals/fixtures/fixture_06_<name>.json` following the schema above.
2. Add a matching mock log to `fetch_test_logs.py` and mock diff to `get_git_diff.py`.
3. Run the harness to establish a baseline score.
4. Tune the relevant prompt in `config/prompts.yaml` if the score is below threshold.
