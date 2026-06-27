# Interview Prep — Q&A

Questions an NVIDIA SDET interviewer is likely to ask about this project,
with answers you can practise out loud. Grouped by topic.

Add new questions here as they come up during mock interviews.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [LangGraph](#2-langgraph)
3. [MCP Server](#3-mcp-server)
4. [Agents](#4-agents)
5. [Evaluation — Custom LLM-as-Judge](#5-evaluation--custom-llm-as-judge)
6. [Evaluation — DeepEval](#6-evaluation--deepeval)
7. [Mock Mode](#7-mock-mode)
8. [CI / GitHub Actions](#8-ci--github-actions)
9. [GPU / NVIDIA Domain Knowledge](#9-gpu--nvidia-domain-knowledge)
10. [Design Decisions](#10-design-decisions)
11. [Tricky Follow-up Questions](#11-tricky-follow-up-questions)

---

## 1. Project Overview

---

**Q: Walk me through what this project does.**

> The AI Triage Agent automates root cause analysis for GPU test failures.
> When a CI test fails, the agent fetches the error log and the git diff,
> runs them through a three-agent LangGraph pipeline — Log Analyzer, Diff
> Analyzer, RCA Synthesizer — and produces a structured report with the root
> cause, recommended fix, priority, and owner. It also scores that report
> using two evaluation judges to verify the quality of the analysis.

---

**Q: Why did you build this? What problem does it solve?**

> At NVIDIA scale, hundreds of GPU tests run every day across training,
> inference, and distributed workloads. When a test fails, an engineer has
> to manually read the log, find the relevant git change, figure out what
> broke, and write up the incident. That takes 30-60 minutes per failure.
> This agent does it in seconds and produces a structured RCA that the
> engineer can act on immediately.

---

**Q: What is the tech stack?**

> Python, LangGraph for multi-agent orchestration, Anthropic Claude as the
> LLM, MCP for tool access, DeepEval for evaluation metrics, Rich for
> terminal reporting, structlog for structured logging, GitHub Actions for CI,
> and pyproject.toml for packaging.

---

**Q: How long did this take to build?**

> The core pipeline — MCP tools, three agents, LangGraph wiring — took about
> a day. The evaluation harness with dual judges, per-fixture scoring, and
> the intentional fail on fixture 4 took another day to get right. The
> trickiest part was the prompt template substitution bug where log content
> wasn't actually flowing into the agent prompts.

---

## 2. LangGraph

---

**Q: What is LangGraph and why did you use it?**

> LangGraph is a Python library for building multi-step AI pipelines where
> each step is a separate agent. Think of it like a traffic police officer
> managing a single-lane road — it controls which agent runs when, carries
> the shared state between them, and stops the flow if any agent signals an
> error. I used it because a single Claude call with everything stuffed into
> one prompt is hard to debug, hard to improve, and gives Claude too much to
> do at once. With LangGraph, each agent has one focused job.

---

**Q: How many agents do you have and what does each one do?**

> Three agents. The Log Analyzer reads the raw error log and extracts
> structured facts — failure type, severity, affected files, stack frames.
> The Diff Analyzer takes those facts plus the git diff and figures out which
> code change caused the failure and how risky it is. The RCA Synthesizer
> combines both outputs into the final report with root cause, fix, priority,
> and owner hint.

---

**Q: How does data flow between agents?**

> Through a shared state object called `TriageState` — it is a Python
> TypedDict. At the start it holds the raw inputs: log text, git diff, commit
> SHA, branch name. Each agent reads what it needs, adds its output, and
> returns the updated state. By the end, the state has everything: the
> original inputs plus log_analysis, diff_analysis, and rca_report all in
> one dictionary.

---

**Q: What happens if one agent fails?**

> After each agent, LangGraph runs a conditional edge function called
> `_has_errors`. It checks whether the errors list in the state is non-empty.
> If there are errors, the graph routes to END immediately — the remaining
> agents are skipped. This prevents a bad log analysis from cascading into
> a meaningless RCA.

---

**Q: Which LangGraph methods did you use?**

> `StateGraph` to create the graph, `add_node` to register each agent,
> `set_entry_point` to declare where to start, `add_conditional_edges` to
> add the error-check branching after each agent, `add_edge` for the
> unconditional final step, `compile` to lock the graph, and `invoke` to
> run the full pipeline and get the final state back.

---

**Q: Is LangGraph installed as a dependency?**

> Yes. It is declared in both `requirements.txt` and `pyproject.toml` as
> `langgraph>=0.1.19`. It gets installed automatically when you run
> `pip install -e .`.

---

## 3. MCP Server

---

**Q: What is MCP and what did you build with it?**

> MCP stands for Model Context Protocol — an open standard for giving AI
> models structured access to external tools, like a USB-C port for AI.
> I built a real MCP server in `src/ai_triage_agent/mcp/server.py` that
> exposes two tools: `fetch_test_logs` and `get_git_diff`. You can start it
> with `python -m ai_triage_agent.mcp.server` and connect any MCP-compatible
> client like Claude Desktop to it.

---

**Q: Is the MCP server called when you run the eval harness?**

> No. The eval harness imports the tool functions directly as Python — no
> server process is involved. The MCP server is a separate entry point for
> when an external AI client like Claude Desktop needs to call those tools
> over the MCP protocol. The underlying tool functions are shared between
> both modes — the server is just a protocol wrapper around the same code.

---

**Q: So the tools are just mock functions you wrote?**

> The MCP server is real — it follows the MCP spec, registers tools with
> JSON schemas, and any MCP client can connect to it. What is mocked is the
> data backend. In production, `fetch_test_logs` would call your CI system
> API and `get_git_diff` would call GitHub. For the demo we pass
> `backend="mock"` and the tools return realistic pre-written data. One
> parameter swap and it works against a real system.

---

## 4. Agents

---

**Q: What is a system prompt and why does it matter?**

> A system prompt is the instruction we give Claude before showing it any
> data. It defines the agent's identity, what it should focus on, and what
> format to return. For example, the Log Analyzer system prompt says "You are
> an expert log analyst. Extract failure signals and return them as JSON with
> these exact fields." Without a precise system prompt, Claude would give a
> freeform answer we can't parse reliably.

---

**Q: Where are the prompts stored and why?**

> All system prompts are stored in `config/prompts.yaml` with version
> numbers. This is an enterprise pattern called prompt versioning. It means
> you can tune an agent's behaviour by editing one YAML file without touching
> any Python code. It also makes it easy to track what changed between
> prompt versions and roll back if a change makes things worse.

---

**Q: What does the Log Analyzer output look like?**

> It returns a JSON object with these fields: `failure_type` (e.g.
> CUDA_OOM), `error_summary`, `stack_frames`, `affected_modules`,
> `severity` (CRITICAL / HIGH / MEDIUM), `reproducible` (true/false), and
> `keywords`. The Diff Analyzer then reads this to know what kind of failure
> to look for in the code change.

---

**Q: What does the final RCA report contain?**

> Title, root_cause, contributing_factors, blast_radius (what else is
> affected), recommended_fix, preventive_measures, priority (P0-P3),
> estimated_fix_time, and owner_hint (which team should fix it).

---

## 5. Evaluation — Custom LLM-as-Judge

---

**Q: How do you evaluate whether the RCA is correct?**

> Two ways. The first is a custom LLM-as-judge: I send the generated RCA and
> the known correct answer to `claude-opus-4-8` and ask it to score the RCA
> on five weighted dimensions — accuracy, actionability, completeness,
> precision, and clarity. Weighted average must be 7.0 or above to pass.
> The second is DeepEval, an industry-standard framework that runs five
> separate metrics including hallucination detection and answer relevancy.

---

**Q: What are the five dimensions in your custom judge?**

> Accuracy at 35% — does the root cause match the ground truth. Actionability
> at 25% — is the fix specific enough to act on immediately. Completeness at
> 20% — are all failure signals addressed. Precision at 10% — no hallucinated
> file names or functions. Clarity at 10% — understandable to a developer
> who didn't write the failing code.

---

**Q: Why does fixture 4 fail?**

> Intentionally. Fixture 4 is a flaky NCCL timeout — a distributed training
> test that fails 15% of the time. Flaky distributed failures are genuinely
> hard to diagnose automatically. The agent's root cause is speculative
> (thermal throttling — not confirmed) and the fix is too vague ("add retry
> logic") for a P2 incident. I wanted the evaluator to be honest rather than
> rubber-stamp everything with a high score. A judge that gives 9/10 to every
> output is not useful.

---

**Q: What is LLM-as-judge?**

> Using a second, usually more powerful LLM to score the output of the first
> LLM. Like having a senior engineer review a junior engineer's report. In
> this project, the triage pipeline uses `claude-sonnet-4-6` to generate the
> RCA, and `claude-opus-4-8` (the more powerful model) acts as the judge and
> scores it. This is a recognised evaluation pattern in AI engineering.

---

## 6. Evaluation — DeepEval

---

**Q: What is DeepEval and why did you add it?**

> DeepEval is an open-source Python library for evaluating LLM outputs with
> standardised metrics. I added it alongside the custom judge to add industry
> credibility. A custom judge is flexible but opaque — someone could question
> whether it is fair. DeepEval uses well-known, published metric definitions
> that the AI evaluation community trusts.

---

**Q: What DeepEval metrics did you use?**

> Five metrics. GEval RCA Correctness — does the root cause match the
> expected failure. GEval Fix Actionability — is the recommended fix specific.
> GEval No Scope Creep — does the RCA stay focused on the actual failure.
> HallucinationMetric — did the agent invent file names or functions that
> don't exist. AnswerRelevancyMetric — does the RCA actually address the
> failure described in the test log.

---

**Q: Why is HallucinationMetric important for RCA specifically?**

> Because a hallucinated file name or function sends engineers on a
> wild-goose chase. If the RCA says "fix line 44 of scaled_dot_product.py"
> but that file doesn't exist, the engineer wastes time searching for it.
> In incident response, every minute counts, so hallucinations are especially
> costly here compared to other use cases.

---

**Q: How do you switch between the custom judge and DeepEval?**

> Using the `--judge` flag when running the harness:
> `--judge custom` runs only the custom LLM-as-judge,
> `--judge deepeval` runs only DeepEval,
> `--judge both` runs both (the default).

---

## 7. Mock Mode

---

**Q: Does this project need an API key to run?**

> No. Set `MOCK_LLM=true` in the `.env` file and the full pipeline runs
> without any API key. Every Claude call is intercepted and returns a
> realistic pre-written response. All five fixtures produce distinct outputs
> with different failure types, scores, and critiques. You can demo the
> entire system end-to-end with zero cost.

---

**Q: How does mock mode know which fixture is running?**

> The mock dispatcher in `llm_client.py` reads the system prompt to detect
> which agent is calling, then reads keywords in the user prompt to detect
> which fixture. For example, if the user prompt contains "NCCL" or
> "allreduce", it returns the TIMEOUT fixture response. If it contains
> "FlashAttentionKernel", it returns the IMPORT_ERROR response. The checks
> are ordered from most specific to least specific to avoid collisions.

---

**Q: How do you switch to real Claude calls?**

> Set `MOCK_LLM=false` and `ANTHROPIC_API_KEY=sk-ant-...` in the `.env`
> file. Everything else stays the same — the same pipeline, the same
> prompts, the same evaluation harness. One variable switches the entire
> system from demo to production.

---

## 8. CI / GitHub Actions

---

**Q: What does your CI pipeline do?**

> Three jobs. The lint job runs Ruff (a fast Python linter) to check code
> style. The test job runs pytest across the full test suite. The
> triage-on-failure job triggers automatically when the test job fails — it
> runs the triage agent on the failed test output and uploads the RCA report
> as a GitHub Actions artifact. So CI doesn't just tell you tests failed —
> it tells you why.

---

**Q: Where is the CI configuration?**

> `.github/workflows/ci.yml`. It defines all three jobs and their triggers.
> The triage job uses a `needs: test` dependency so it only runs after the
> test job, and an `if: failure()` condition so it only triggers on actual
> test failures.

---

## 9. GPU / NVIDIA Domain Knowledge

---

**Q: What is CUDA OOM and why does it happen?**

> CUDA Out of Memory. When a GPU training job requests more memory than the
> GPU has available, PyTorch throws a RuntimeError. Common causes: batch
> size too large, model too big for the GPU, memory leak in training loop.
> In our fixture, the batch size was increased 8x (from 8 to 64) which
> pushed memory usage from ~1.5 GiB to ~14 GiB — more than the GPU had free
> after loading model weights.

---

**Q: What is NCCL?**

> NVIDIA Collective Communications Library. It coordinates multiple GPUs
> during distributed training. Operations like `allreduce` require every GPU
> to share gradients with every other GPU and all receive the combined sum.
> NCCL manages this synchronisation. If one GPU is slow or goes offline,
> the others wait — which causes timeouts in our fixture 4 scenario.

---

**Q: What is BF16 and why does it require a specific CUDA version?**

> BF16 stands for Brain Float 16 — a 16-bit number format used in AI
> training. It uses less GPU memory than 32-bit floats but has a wide enough
> range to train most models without instability. Tensor core hardware support
> for BF16 was added in CUDA 12.1. If your CI runner has CUDA 11.8, any code
> that calls `torch.cuda.is_bf16_supported()` or asserts a minimum CUDA
> version will fail immediately.

---

**Q: What is gradient accumulation?**

> A technique to simulate a large batch size without using more GPU memory.
> Instead of processing 64 samples in one batch, you process 8 samples at a
> time over 8 steps and accumulate the gradients before updating the weights.
> The mathematical result is identical to a single batch of 64, but peak
> memory usage stays at the 8-sample level. This is the recommended fix for
> the CUDA OOM fixture.

---

## 10. Design Decisions

---

**Q: Why three agents instead of one?**

> Each agent has one focused job, its own system prompt, and produces
> structured JSON that the next agent can reliably read. One big prompt is
> harder to debug — if the RCA is wrong, you don't know whether the problem
> was in reading the log, understanding the diff, or synthesising the report.
> Three agents give you three intermediate checkpoints you can inspect.

---

**Q: Why store prompts in a YAML file instead of hardcoding them?**

> Prompt tuning is an ongoing activity. If you hardcode prompts in Python,
> every change requires a code review, a merge, and a deploy. With prompts
> in a versioned YAML file, a non-engineer can improve them with a text
> editor. It also makes it easy to run A/B tests between prompt versions.

---

**Q: Why did you add an intentional FAIL in the eval suite?**

> To prove the evaluator is honest. If all five fixtures score 9/10, an
> interviewer might reasonably suspect the scores are fabricated or that the
> judge is too lenient. The TIMEOUT fixture genuinely deserves a lower score
> because flaky distributed failures are hard to diagnose automatically — the
> root cause is speculative and the fix is vague. Showing a 6.7 FAIL
> demonstrates the system has real discriminating power.

---

**Q: Why use two judges instead of one?**

> The custom judge is flexible and tailored to our RCA quality criteria, but
> it is opaque — built by us, scoring by us. DeepEval adds independent
> credibility using published metrics the AI community recognises.
> Running both means we catch different failure modes: the custom judge
> catches vague fixes, DeepEval catches hallucinations and off-topic answers.

---

## 11. Tricky Follow-up Questions

---

**Q: If I ask Claude for an RCA and it gives me a wrong answer, how would you know?**

> That is exactly what the eval harness tests. Each fixture has a
> `ground_truth` field with the correct RCA. The custom judge compares the
> generated RCA against the ground truth on five dimensions and flags
> anything below 7.0. DeepEval's HallucinationMetric catches invented facts.
> In production you would run this on every triage output and alert if scores
> drop below threshold.

---

**Q: What would you need to change to make this work against real CI data?**

> Two things. First, update the data backends in `fetch_test_logs` and
> `get_git_diff` — swap `backend="mock"` for `backend="github_actions"` or
> `backend="github_api"` and add the real API credentials. Second, set
> `MOCK_LLM=false` and provide a real Anthropic API key. The pipeline,
> agents, prompts, and evaluation harness all stay exactly the same.

---

**Q: How would you scale this to run on thousands of test failures per day?**

> Replace the synchronous `triage_graph.invoke()` with an async queue.
> Each CI failure pushes a job to the queue. A pool of workers picks up jobs
> and runs the triage pipeline. Results are written to a database. The MCP
> server already supports async via Python's asyncio. LangGraph also has
> async support with `.ainvoke()` — one method name change.

---

**Q: What was the hardest bug you hit building this?**

> The prompt template substitution bug. All five fixtures were returning
> identical scores and critiques even though the fixtures had completely
> different failure types. Turned out the prompt loader was using Python's
> `string.Template` which expects `$variable` syntax, but our YAML prompts
> used `{variable}` syntax. The log content was never being substituted into
> the user prompt — Claude was seeing the literal text `{log_content}` and
> falling back to the same generic response every time. Fixing it to use
> `str.format_map` with a safe dict immediately made all five fixtures
> produce distinct outputs.

---

**Q: Could you replace LangGraph with something simpler — just calling three functions in sequence?**

> Yes, for this pipeline you could. Three function calls in sequence would
> work. LangGraph adds value when you need branching, loops, parallel
> execution, or checkpointing across many agents. For our three-step linear
> pipeline the main benefits are: the conditional edge error handling, the
> typed state enforced by `TriageState`, and the clean separation of graph
> structure from agent logic. If the pipeline grows to 10 agents with
> branching, LangGraph becomes essential.

---

**Q: Why Claude and not GPT-4 or another model?**

> This is a NVIDIA SDET role. Anthropic's Claude models score well on
> technical reasoning and code understanding benchmarks, and claude-opus-4-8
> is particularly strong at following structured output instructions reliably —
> which matters because each agent must return parseable JSON. The project
> is also model-agnostic at the architecture level — swapping to GPT-4 would
> require changing one line in `llm_client.py`.

---
