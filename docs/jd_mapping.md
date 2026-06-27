# NVIDIA JD — Line by Line Breakdown

Every sentence from the job description, explained in plain English,
mapped to exactly what this project demonstrates, and with a talking point
you can use in the interview.

---

## Part 1 — The Vision (Opening Paragraph)

---

### "Build infrastructure that thinks — that triages failures, files bugs, and surfaces root causes without waiting for a human to ask."

**What it means:**
Traditional test automation tells you *that* something failed. NVIDIA wants
infrastructure that tells you *why* it failed, *what to do about it*, and
*who should fix it* — all automatically, before any engineer even opens a
terminal.

**How your project maps:**
This is exactly what the AI Triage Agent does. When a CI test fails, it does
not wait for an engineer. It automatically fetches the log, fetches the diff,
runs three agents in sequence, and produces a structured RCA with root cause,
recommended fix, priority, and owner hint.

**Talking point:**
> "That sentence is literally the one-line description of my project. The
> agent triages the failure, surfaces the root cause, and tells you who owns
> the fix — without a human asking. I built it end-to-end with three agents,
> an MCP server for data access, and an eval harness to verify the quality
> of the output."

---

### "A small group of high-agency engineers, equipped with well-designed autonomous agents, can accomplish what previously required a much larger organisation."

**What it means:**
"High-agency" means engineers who take ownership and act without being told
every step. The idea is: give one senior engineer the right AI tools and they
can do the work of a 10-person triage team. NVIDIA wants to build that
multiplier effect into their infrastructure.

**How your project maps:**
The eval harness runs 5 fixture scenarios end-to-end and produces a full
quality report — work that would otherwise require a QA engineer to manually
read 5 logs, write 5 RCAs, and peer-review each one. One command replaces
that entire workflow.

**Talking point:**
> "High-agency to me means: you understand the problem deeply enough to build
> the right tool, and you build it so it runs without you. The harness runs
> unattended, scores every output, and saves a JSON report. An engineer
> doesn't need to be in the room for it to work."

---

### "This isn't about using AI tools to work faster — it's about building the infrastructure that other engineers depend on."

**What it means:**
They do not want someone who uses GitHub Copilot to write code faster.
They want someone who *builds the systems* that entire teams rely on — the
plumbing under the platform, not productivity tricks on top of it.

**How your project maps:**
The project is infrastructure: a pipeline other engineers would plug their
CI failures into. The MCP server exposes tools any AI client can call. The
eval harness is a quality gate other teams could run against their own agents.
The prompts.yaml file means non-engineers can tune the agents without touching
Python.

**Talking point:**
> "The distinction I kept in mind while building this: if I got hit by a bus,
> could another engineer pick this up? That is why the docs folder exists,
> why prompts are in a YAML file, and why mock mode means anyone can run the
> full pipeline without an API key."

---

## Part 2 — What You'll Be Doing

---

### "Develop and deploy multi-agent systems for automated test generation, log analysis, failure triage, and bug-filing workflows."

**What it means — term by term:**

| Term | Plain English |
|------|--------------|
| Multi-agent systems | Multiple specialised AI agents, each with one job, working together |
| Automated test generation | AI writes the test cases, not a human |
| Log analysis | AI reads error logs and extracts structured facts |
| Failure triage | AI classifies what broke, how severe, who owns it |
| Bug-filing workflows | AI creates the JIRA/GitHub ticket with the RCA pre-filled |

**How your project maps:**
- **Multi-agent:** Three agents — Log Analyzer, Diff Analyzer, RCA Synthesizer
- **Log analysis:** Log Analyzer agent reads raw CI error logs
- **Failure triage:** RCA Synthesizer assigns priority (P0-P3) and owner hint
- **Test generation / Bug-filing:** Not built yet — honest gap to acknowledge

**Talking point:**
> "My project covers log analysis, failure triage, and root cause surfacing.
> Bug-filing would be the natural next step — the RCA report already has
> everything a JIRA ticket needs: title, description, priority, owner. Wiring
> that to a JIRA MCP tool is a one-agent addition to the existing pipeline."

---

### "Build and maintain agent orchestration frameworks using tools such as Claude Code, MCP servers, and agent SDK patterns."

**What it means — term by term:**

| Term | Plain English |
|------|--------------|
| Agent orchestration framework | The system that decides which agent runs when, in what order, with what data |
| Claude Code | Anthropic's AI coding assistant (the tool you are using right now) |
| MCP servers | Servers that expose tools to AI models via the Model Context Protocol |
| Agent SDK patterns | Standard patterns for building agents: tool calling, state management, error handling |

**How your project maps:**
- **Orchestration framework:** LangGraph in `graph/workflow.py` — StateGraph, nodes, conditional edges
- **MCP server:** `src/ai_triage_agent/mcp/server.py` — real MCP server with two tools
- **Agent SDK patterns:** Each agent follows the same pattern: read state → call LLM → write state → return

**Talking point:**
> "I built the orchestration layer with LangGraph — it manages agent sequencing,
> shared state, and error routing. The MCP server follows the open standard so
> any MCP-compatible client can call the tools. The agent pattern is consistent
> across all three agents — same structure, different prompts — which makes it
> easy to add a fourth agent without changing the framework."

---

### "Create autonomous pipelines that reduce cognitive load on engineers by routing failures, surfacing root causes, and generating actionable bug reports."

**What it means:**
"Cognitive load" = the mental effort an engineer has to spend. Reading a 500-
line error log is high cognitive load. Reading a three-sentence RCA that says
"BATCH_SIZE went from 8 to 64, use gradient accumulation, fix time 2 hours,
owner: ml-training team" is low cognitive load. The pipeline does the hard
thinking so the engineer just acts.

**How your project maps:**
The RCA Synthesizer produces: title, root_cause, contributing_factors,
blast_radius, recommended_fix, preventive_measures, priority, estimated_fix_time,
owner_hint. An engineer reads this and knows exactly what to do — no log
reading required.

**Talking point:**
> "Cognitive load reduction is measurable. Before: engineer reads 500-line log,
> finds the relevant diff, writes the RCA — 45 minutes. After: agent produces
> a structured report in under 10 seconds. The engineer spends 2 minutes
> reviewing and 5 minutes acting. That is the multiplier."

---

### "Build evaluation systems to measure agent output quality — ensuring autonomous pipelines are reliable, not just fast."

**What it means:**
Fast is not enough. If the agent produces wrong RCAs quickly, it is worse than
useless — engineers will chase false leads. You need a system that continuously
checks: is the agent still producing correct, actionable output? This is called
an evaluation (eval) framework.

**How your project maps:**
This is the dual eval harness. Custom LLM-as-judge scores on 5 weighted
dimensions with a 7.0 pass threshold. DeepEval adds HallucinationMetric,
AnswerRelevancyMetric, and three GEval metrics. Both run against 5 fixtures
with known ground truth answers. Results saved as JSON reports.

**Talking point:**
> "I built a dual eval harness — a custom LLM-as-judge for our specific RCA
> quality criteria, plus DeepEval for industry-standard metrics including
> hallucination detection. The intentional fail on fixture 4 proves the
> evaluator is honest. In production, this harness would run on every deploy
> to catch prompt regressions before they hit real failures."

---

### "Establish observability and monitoring for agentic workflows so failures are transparent, debug-gable, and recoverable."

**What it means:**

| Term | Plain English |
|------|--------------|
| Observability | Can you see what the agent is doing at every step, not just the final output? |
| Monitoring | Is the agent healthy right now? Are scores dropping over time? |
| Transparent | When something goes wrong, you can see exactly where and why |
| Debug-gable | You can reproduce the failure and trace it to the exact cause |
| Recoverable | The system handles failures gracefully — it does not silently produce wrong output |

**How your project maps:**
- **Observability:** structlog in every agent — every LLM call is logged with agent name, fixture, failure type, elapsed time
- **Transparent failures:** Conditional edges write to the `errors` list in state — any downstream agent can see what went wrong upstream
- **Recoverable:** `_has_errors()` short-circuits the pipeline rather than letting errors cascade silently
- **JSON reports:** Every eval run saves a timestamped JSON to `evals/reports/` — you can compare runs over time

**Talking point:**
> "Every agent logs structured events with structlog — you can see exactly
> which agent ran, what it detected, and how long it took. Errors are written
> to the shared state so no agent runs on bad input silently. The eval reports
> are timestamped JSON — you can plot scores over time and detect when a
> prompt change degraded quality."

---

### "Build internal tooling that is adoptable, not just technically impressive — with clear documentation and low onboarding friction."

**What it means:**
Engineers often build clever systems that only they understand. NVIDIA wants
tools that other engineers can pick up, run, and extend without asking the
original author. Low onboarding friction = someone new can get it running in
under 10 minutes.

**How your project maps:**
- `MOCK_LLM=true` means zero setup — no API key, no external service, runs immediately
- `docs/walkthrough.md` explains every term, every file, every command
- `docs/interview_prep.md` (this file) documents the design decisions
- `config/prompts.yaml` means non-engineers can tune agents without Python knowledge
- `pip install -e .` installs everything in one command

**Talking point:**
> "I specifically designed for adoptability. Mock mode means a new engineer
> can clone the repo and run the full pipeline in under 2 minutes without any
> credentials. The docs folder has a walkthrough that explains every term
> from scratch. Prompts are in YAML so someone who doesn't write Python can
> still improve the agents."

---

## Part 3 — What We Need to See

---

### "Strong Python engineering — clean, testable, maintainable code with a systems-level perspective."

**What it means:**
- **Clean:** Easy to read, well-named, no hacks
- **Testable:** Functions are small and focused, dependencies are injectable
- **Maintainable:** Someone else can change it six months later without breaking everything
- **Systems-level:** You think about how pieces fit together, not just individual functions

**How your project maps:**
- Each agent is a single focused function in its own file
- `TriageState` is a typed TypedDict — no untyped dictionaries
- `pyproject.toml` with Ruff linter and mypy type checker enforces quality
- MCP tools are in `mcp/tools/` — shared between the server and the harness, no duplication
- Tests in `tests/` with pytest

**Talking point:**
> "Every agent is a pure function: takes state, returns state. No global
> variables, no side effects outside the state. That makes them individually
> testable. Ruff and mypy run in CI to enforce code quality automatically."

---

### "Deep familiarity with AI-native development workflows — Claude Code, Cursor, LLM APIs, prompt engineering in production."

**What it means:**

| Term | Plain English |
|------|--------------|
| AI-native development | You use AI tools as a core part of how you write code, not as an afterthought |
| Claude Code | Anthropic's AI coding assistant in the terminal (what you are using now) |
| LLM APIs | Calling language models programmatically — sending prompts, receiving structured responses |
| Prompt engineering in production | Writing and maintaining prompts that reliably produce parseable, correct output at scale |

**How your project maps:**
- Built using Claude Code throughout
- Direct Anthropic API calls in `llm_client.py` with `call_llm_json()`
- Versioned prompts in `config/prompts.yaml` — production prompt engineering pattern
- `str.format_map` with `_SafeDict` for safe variable substitution in prompts
- System prompts tuned to return strict JSON that agents can parse reliably

**Talking point:**
> "Prompt engineering in production is different from experimenting in a
> playground. You need prompts that return parseable JSON every time, handle
> edge cases gracefully, and can be improved without breaking downstream code.
> That is why prompts are versioned in YAML and why each system prompt
> specifies the exact JSON schema the agent expects."

---

### "Hands-on experience building multi-agent or autonomous systems that have shipped and run without continuous supervision."

**What it means:**
Not a demo or a notebook. A system that runs unattended, handles its own
errors, and does not need someone watching it to work correctly.

**How your project maps:**
- The eval harness runs all 5 fixtures end-to-end with no human input
- Conditional edges handle agent failures automatically — no supervision needed
- GitHub Actions CI runs the triage pipeline automatically on test failure
- JSON reports are saved automatically — no one needs to be present

**Talking point:**
> "The harness runs completely unattended. You trigger it once and walk away.
> If an agent fails, the pipeline short-circuits and writes the error to the
> report rather than crashing or producing silent garbage. GitHub Actions
> runs it automatically on every test failure — no one needs to be watching."

---

### "Clear understanding of where LLMs fail — hallucination, context degradation, tool misuse — and experience building mitigations."

**What it means:**

| LLM failure mode | Plain English |
|-----------------|--------------|
| Hallucination | The LLM invents facts — file names, function names, error messages that don't exist |
| Context degradation | With very long inputs, LLMs start ignoring earlier content (the "lost in the middle" problem) |
| Tool misuse | The LLM calls a tool with wrong arguments or at the wrong time |

**Mitigations = how you prevent or catch these:**

| Failure | Mitigation in this project |
|---------|--------------------------|
| Hallucination | HallucinationMetric in DeepEval catches invented file names. Precision dimension (10%) in custom judge penalises hallucinated functions |
| Context degradation | Each agent gets a focused, short prompt — only the data it needs, not the full history. Log Analyzer does not see the git diff. |
| Tool misuse | MCP tools have JSON schemas with required fields and enum constraints — invalid calls are rejected at the schema level |
| Wrong JSON output | `call_llm_json()` strips markdown fences and parses JSON — handles common LLM formatting mistakes |
| Silent failures | Conditional edges write errors to state and short-circuit — never silently continue on bad input |

**Talking point:**
> "I built two explicit mitigations for hallucination: the Precision dimension
> in the custom judge penalises invented file names, and DeepEval's
> HallucinationMetric measures the hallucination rate directly. For context
> degradation, each agent only receives the data it needs — the Log Analyzer
> never sees the git diff, so it cannot get confused by it. For tool misuse,
> MCP tool schemas enforce required fields and valid enum values at the
> protocol level."

---

## Part 4 — Ways to Stand Out

---

### "Built and shipped MCP servers, custom tool integrations, or multi-agent orchestrations — with working examples to show."

**Your position:**
You have all three. Working MCP server in `mcp/server.py`. Custom tool
integrations — `fetch_test_logs` and `get_git_diff` with swappable backends.
Multi-agent orchestration with LangGraph. And it runs — `python -m
evals.harness --judge both` produces a real report right now.

**Talking point:**
> "I can show all three running right now. The MCP server starts with one
> command. The multi-agent pipeline runs end-to-end with mock mode — no API
> key needed for the demo. The eval report shows per-fixture scores with
> distinct critiques. It is not a prototype — it is a complete working system."

---

### "Designed evaluation harnesses or scoring systems that measure and enforce LLM output quality at scale."

**Your position:**
Dual harness: custom 5-dimension weighted judge + DeepEval with 5 metrics.
Per-fixture scoring. Timestamped JSON reports. Pass/fail threshold enforcement.
Intentional fail on fixture 4 proving the system is honest.

**Talking point:**
> "The harness does not just run — it enforces quality. There is a hard pass
> threshold of 7.0. Fixture 4 intentionally fails at 6.7 because the root
> cause is speculative and the fix is too vague. A system that passes
> everything is not an evaluation system — it is a rubber stamp."

---

### "Built agentic systems with graceful failure recovery — retry logic, fallback chains, human-in-the-loop escalation."

**What it means:**

| Term | Plain English |
|------|--------------|
| Retry logic | If the LLM call fails (network error, rate limit), try again automatically |
| Fallback chains | If Agent A fails, route to a simpler fallback Agent B instead of crashing |
| Human-in-the-loop escalation | If the agent is not confident, flag it for a human to review rather than publishing a wrong answer |

**How your project maps:**
- **Retry logic:** `tenacity` in `llm_client.py` — 3 retries with exponential backoff on LLM API failures
- **Graceful failure:** Conditional edges write errors to state and stop the pipeline cleanly — no crash, no silent wrong output
- **Human-in-the-loop:** The priority field and the eval score together signal when to escalate. A P0 failure with a 6.7 judge score is a clear signal that a human should review the RCA before acting.

**Honest gap — fallback chains:**
A full fallback chain (e.g. if Claude fails, try a smaller model) is not built.
Be honest: "Retry logic is implemented with tenacity. A full fallback chain
to a secondary model is the natural next step."

**Talking point:**
> "Retry logic is in from day one — tenacity handles transient API failures
> with exponential backoff. Graceful failure is built into the graph — if any
> agent errors, the pipeline stops cleanly and writes a readable error to the
> report rather than crashing or producing garbage. The combination of
> priority level and eval score creates a natural human-in-the-loop signal:
> if the agent is not confident, the score reflects that and a human reviews
> before acting."

---

### "Experience with NVIDIA Omniverse, OpenUSD, or similarly complex platform SDKs."

**What it means:**

| Term | Plain English |
|------|--------------|
| NVIDIA Omniverse | NVIDIA's platform for 3D simulation and collaboration — used in robotics, autonomous vehicles, digital twins |
| OpenUSD | Universal Scene Description — an open format for 3D scenes, originally from Pixar, now widely used in Omniverse |
| Complex platform SDK | A large, multi-component software platform where testing is non-trivial — lots of moving parts, GPU dependencies, version sensitivity |

**Your position:**
You do not have direct Omniverse experience. Be honest. But the triage agent
directly addresses the *type of problem* Omniverse testing creates: GPU memory
failures, CUDA version mismatches, distributed system timeouts — all of which
appear in your 5 fixtures.

**Talking point:**
> "I don't have direct Omniverse experience, but the failure types I modelled
> in this project are exactly what you see in GPU-heavy platforms: CUDA OOM,
> environment mismatches, distributed training timeouts, performance
> regressions. The triage infrastructure is not Omniverse-specific — it is
> designed to analyse any structured test failure, and I would ramp up on
> Omniverse's specific failure patterns quickly."

---

### "Can point to infrastructure you've shipped that measurably reduced a team's manual triage burden — with clear documentation."

**Your position:**
This is a portfolio project built for this interview, not production-shipped.
Be honest about that, but frame what it *demonstrates*:

**Talking point:**
> "This is a portfolio project built to demonstrate these exact capabilities,
> not a production deployment. What I can show: the infrastructure is complete,
> documented, and runnable by anyone in under 2 minutes. The design decisions
> — versioned prompts, mock mode, dual eval harness, structured state — are
> the same ones I would apply to a production deployment. The gap between
> this and production is data backends and API credentials, not architecture."

---

## Summary — How to Open the Interview

If they ask "tell me about a project you're proud of":

> "I built an AI Triage Agent for automated test failure analysis — the kind
> of infrastructure this role describes. When a CI test fails, the agent
> fetches the error log and git diff through an MCP server I built, runs them
> through a three-agent LangGraph pipeline — Log Analyzer, Diff Analyzer, RCA
> Synthesizer — and produces a structured report with root cause, fix,
> priority, and owner.
>
> I also built a dual evaluation harness to measure output quality: a custom
> LLM-as-judge scoring five weighted dimensions, plus DeepEval for
> hallucination detection and answer relevancy. One of the five test fixtures
> intentionally fails — a flaky distributed timeout — to prove the evaluator
> is honest, not a rubber stamp.
>
> The whole thing runs in mock mode with no API key. I can show it running
> right now."

That opening covers: MCP, multi-agent, LangGraph, evaluation, hallucination
mitigation, graceful failure, and adoptability — every major requirement in
the JD — in under 60 seconds.
