# Code Checkpoint

**CIS6930 — Spring 2026**
**Due: March 26, 2026**

---

## Team Members

| Name | Role |
|------|------|
| Sanjeev Kamath | Solo — full-stack implementation: data extraction, LLM orchestration, schema validation, persistence, stability analysis, testing |

---

## Response to Proposal Feedback

### Q1: How was the 50% volatility reduction threshold chosen, and what would be the practical interpretation of a result that falls short?

The 50% threshold was chosen to represent a *practically meaningful* improvement, not just a statistically detectable one. The metric is:

```
volatility_reduction_pct = (constrained_mean - unconstrained_mean) / (1 - unconstrained_mean)
```

This formula measures reduction relative to the existing volatility — so 50% means the constrained condition closes half the gap between the unconstrained baseline and perfect stability (Jaccard = 1.0). Concretely: if unconstrained runs produce a mean row-level Jaccard of 0.60 (40% of token content varies across runs), a 50% reduction would bring constrained runs to Jaccard ≈ 0.80 (only 20% varies). That's a substantive improvement for a production pipeline where you'd be storing these records and comparing them over time.

The threshold was set at 50% rather than, say, 20% because the schema constraint mechanism is fairly aggressive — it locks down `risk_level` (integer 1–4), `risk_label` (enum of 4 values), and `event_types` (enum of 8 values) and retries with structured error feedback. Fields like these should converge quickly. A 20% improvement might just reflect that the LLM happens to get structured fields right most of the time even without constraints.

**If the result falls short (e.g., 35–45%):** that's still interpretable as partial stabilization and worth reporting. It would likely mean:
- The schema constraints successfully anchor the *structural* fields (`risk_level`, `risk_label`, `event_types`) — which are enumerable and easy to validate
- But the *semantic* fields (`advisory_summary`, `regional_warnings`, `entry_exit_requirements`) vary naturally because they require synthesis and no amount of schema enforcement prevents the LLM from paraphrasing differently on each run
- The practical implication would be: schema constraints are necessary but not sufficient for full stability; downstream applications that depend on free-text summaries would still need human review or additional normalization

A 35% result would not invalidate the study — it would answer a more nuanced version of the question. The threshold is a benchmark, not a pass/fail cutoff.

### Q2: What prompt engineering strategy is used, and how is prompt design held constant across conditions?

The prompt strategy is deliberately minimal and frozen. The full `SYSTEM_PROMPT` is defined as a **module-level constant** in `src/pipeline/orchestrator.py` and is never modified between runs or conditions — it's the same Python object used in every call. Here's what it does:

1. **Role framing:** "You are a travel risk analyst integrating data from the U.S. State Department and recent news articles." Establishes the synthesis task without over-constraining the output.

2. **Explicit field rules inline:** The system prompt lists every required field and its constraint (`risk_level: integer 1–4 only`, `event_types: non-empty array; each item must be exactly one of: crime, terrorism, ...`). This is drawn from the schema but expressed in plain English so the LLM understands it before generation, not just as post-hoc validation.

3. **Anchored fields:** Six fields (`run_id`, `timestamp`, `country_name`, `country_code`, `risk_level`, `risk_label`) are passed in the user prompt as literal values the LLM is told to include unchanged. This prevents the LLM from hallucinating the risk level or country code, and ensures the State Department's own assessment is preserved rather than reinterpreted.

4. **Output format instruction:** "Output ONLY valid JSON. No markdown, no code fences, no explanation before or after." This reduces the incidence of wrapper text that would fail parsing.

**How the conditions are held constant:** The constrained and unconstrained conditions use the exact same `SYSTEM_PROMPT`, the same `_build_user_prompt()` function, the same temperature (1.0), and the same model. The *only* difference is what happens after the LLM responds:
- **Constrained:** run `validate_record()` → if invalid, append the validation errors to the conversation and retry (up to 3 times)
- **Unconstrained:** accept whatever the LLM returns, no retry

No prompt tuning is done separately for each condition. If the prompt were different, any observed stability difference could be attributed to the prompt rather than the constraint mechanism — that would defeat the study. The frozen `SYSTEM_PROMPT` constant is the implementation-level guarantee that this doesn't happen.

### On confidence intervals

Per the feedback, bootstrap 95% CIs are now computed for every stability metric. The `_bootstrap_ci()` function in `stability_analysis.py` samples with replacement 1000 times (seed=42 for reproducibility) and returns the 2.5th/97.5th percentile. All four tools (`compare_runs`, `compute_stability_report`, `compare_conditions`, `get_field_stability`) report CIs on every aggregate metric. With 30 countries and the number of pairs from 5+ runs, the CI width should be informative.

---

## What Has Been Completed

### Data Extraction

**`src/servers/state_dept_extract.py`** — Fetches travel advisories from the U.S. State Department RSS feed, parses advisory level and text from the detail pages, and caches results as JSON. All 30 target countries are cached at `data/raw/state_dept/{CODE}.json`. The server is idempotent — re-running skips already-cached countries. Risk distribution across the 30 countries: 6 × Level 1, 12 × Level 2, 6 × Level 3, 6 × Level 4.

**`src/servers/news_extract.py`** — Fetches one recent news article per country from NewsAPI and caches results at `data/raw/news/{CODE}.json`. All 30 articles are cached. Future pipeline runs cost zero additional API calls. Both servers expose their data via MCP tools callable from the orchestrator.

### Schema and Validation

**`src/schema/travel_advisory.json`** — 12-field strict JSON Schema (Draft 7). Key constraints: `risk_level` integer 1–4, `risk_label` enum of 4 State Dept labels, `event_types` items from an 8-value taxonomy, `country_code` pattern `^[A-Z]{3}$`, `additionalProperties: false` rejects any field the LLM invents. The schema is the ground truth for what a valid record looks like.

**`src/servers/schema_validation.py`** — Validates LLM output against the schema using `jsonschema.Draft7Validator`. On failure, generates a structured retry prompt that lists every offending field path and error message. Tested against a deliberately broken 11-error record — all errors caught in a single pass. Also exposes `validate_batch()` for batch validation and `get_field_constraints()` which returns a concise constraint summary suitable for injection into prompts.

### LLM Orchestration

**`src/pipeline/orchestrator.py`** — The core of the experiment. `Orchestrator(constrained=True)` applies schema validation and a multi-turn retry loop (up to 3 attempts); `Orchestrator(constrained=False)` is the single-shot baseline. Both use the identical frozen `SYSTEM_PROMPT`, identical temperature (1.0), and identical cached input data. The retry loop appends the validation error feedback as a new user message so the LLM sees exactly what it needs to fix. Verified working against the live `gpt-oss-120b` endpoint in a two-country pilot (FRA, JPN — both succeeded on first attempt, schema pass rate 1.0).

### Persistence

**`src/servers/load_server.py`** — SQLite-backed persistence at `data/travel_advisory.db`. Inserts validated records with upsert semantics (safe to re-run). Creates versioned run snapshots that store run-level metadata: `constrained` flag, `total`/`succeeded`/`failed` counts, `schema_pass_rate`, `avg_attempts`. Exposes `insert_run_results()` to persist an entire pipeline output in one call. Two runs are stored from the checkpoint pilot.

### Stability Analysis

**`src/servers/stability_analysis.py`** — Four MCP tools for the full evaluation:

- `compare_runs(run_id_a, run_id_b)` — pairwise comparison with per-country breakdown and bootstrap CIs
- `compute_stability_report(run_ids)` — all C(N,2) pairs within a run set, CI over pair means
- `compare_conditions(constrained_run_ids, unconstrained_run_ids)` — the main research claim tool; computes `volatility_reduction_pct` with hypothesis test
- `get_field_stability(run_ids, field)` — single-field drill-down (exploratory)

Primary metric: **row-level Jaccard similarity** — each record is converted to a typed token set (`risk_level:2`, `event_type:crime`, `regional_warning:Paris suburbs`, etc.) and Jaccard similarity is computed between runs. Missing countries score 0.0, anchoring the metric to the full expected 30-country set. Secondary metrics: reproducibility rate, entity stability (exact-match for categorical fields, set Jaccard for arrays), and cosine similarity of `advisory_summary` embeddings using `all-MiniLM-L6-v2`.

### Tests

| File | Tests | What's covered |
|------|-------|---------------|
| `tests/test_servers.py` | 94 | Schema validation edge cases (13), batch validation (4), load server upsert/idempotency/retrieval (17), stability analysis helpers (25), all MCP tool integration paths including error handling (35) |
| `tests/test_pipeline.py` | 47 | JSON parsing (8), prompt construction (7), orchestrator construction (5), constrained retry loop (7), unconstrained single-shot (4), edge cases (3), pipeline summary stats (13) |
| **Total** | **141** | All pass. No API calls, no model downloads — OpenAI client and sentence-transformer encoder are mocked. |

---

## What Is In Progress

**Running the full experiment.** The infrastructure is complete — the blocker is simply wall-clock time. Each full 30-country pipeline run takes several minutes due to LLM latency and the per-request delay (1 second between API calls to avoid hammering the HiPerGator endpoint). The plan is 5 constrained + 5 unconstrained runs, stored under separate run IDs, then `compare_conditions()` to compute the volatility reduction.

**Documentation.** `results/preliminary_results.md` and `docs/progress_report.md` are being finalized alongside this checkpoint.

---

## What Remains To Be Done

| Task | Notes |
|------|-------|
| Run full experiment (5× constrained, 5× unconstrained) | Core deliverable for the draft paper; all code is ready |
| Compute and report stability metrics | One `compare_conditions()` call once runs are stored |
| Investigate per-risk-level variance | Do Level 4 countries (AFG, SYR, YEM) show higher instability? |
| Annotate a ground truth subset | 10-country human annotation for accuracy check (proposal commitment) |
| Write draft paper | Analysis, discussion of results, limitations |
| Final paper and presentation | April deadlines |

The scope is manageable. The hard implementation work is done — what remains is running the experiment and writing it up.

---

## How To Run

### Prerequisites

- Python 3.13+, [`uv`](https://docs.astral.sh/uv/)
- A `.env` file in the project root:
  ```
  NAVIGATOR_API_KEY=your-key-here
  ```

### Install

```bash
uv sync
```

### Run the pipeline

```bash
# Constrained mode (schema validation + retry loop)
uv run python -m src.pipeline.orchestrator

# Unconstrained baseline
uv run python -m src.pipeline.orchestrator --unconstrained

# Single country (good for testing)
uv run python -m src.pipeline.orchestrator --country FRA

# Save output to file
uv run python -m src.pipeline.orchestrator --output results/run_001.json
```

Raw data (30 advisories + 30 news articles) is fully cached in `data/raw/` — no internet access required after cloning.

### Persist results and compare runs

*To be implemented post-checkpoint.* Once the full experiment runs are complete, results will be loaded into SQLite via `load_server.insert_run_results()` and compared using `stability_analysis.compare_conditions()`. A dedicated run script will be added at that stage.

### Run tests

```bash
uv run pytest               # all 141 tests
uv run pytest tests/test_servers.py   # MCP servers only
uv run pytest tests/test_pipeline.py  # orchestrator only
```

No API key needed for tests — everything is mocked.