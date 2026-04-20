# CIS6930 Project — Travel Risk Assessment Stability Analysis

**Can schema constraints make an LLM more consistent?**

This project builds a multi-source data pipeline that synthesizes U.S. State Department travel advisories and news articles into structured travel risk records using an LLM. The core research question: does adding schema validation and a retry loop reduce output volatility by at least 50% compared to single-shot unconstrained generation — measured across 30 countries and repeated runs?

Two models are tested: `gpt-oss-120b` (120B-parameter LLM) and `llama-3.1-8b-instruct` (8B-parameter SLM), both running on UF's HiPerGator cluster. Temperature is intentionally set to 1.0 to measure natural variance — setting it to 0 would trivially stabilize outputs and defeat the point.

---

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) — used for everything (`uv sync`, `uv run`)
- A `NAVIGATOR_API_KEY` for the UF HiPerGator LLM endpoint

The raw data (30 state dept advisories + 30 news articles) is already cached in `data/raw/`. You don't need NewsAPI or internet access to run the pipeline.

---

## Setup

```bash
git clone https://github.com/sanjeevkamath/cis6930sp26-project.git
cd cis6930sp26-project
uv sync
```

Create a `.env` file in the project root:

```
NEWSAPI_KEY=''
NAVIGATOR_API_KEY='sk-.....'
OPENAI_BASE_URL=https://api.ai.it.ufl.edu/v1
OPENAI_MODEL=gpt-oss-120b
```

> **Switching models:** To run with the smaller Llama model, change `OPENAI_MODEL` in your `.env`:
> ```
> OPENAI_MODEL=llama-3.1-8b-instruct
> ```
> No other changes are needed — the pipeline, schema, and prompts are identical for both models.

---

## Running the Pipeline

### Constrained mode (schema validation + retry loop)

```bash
uv run python -m src.pipeline.orchestrator
```

This runs all 30 countries with up to 3 retries per country. Each failed validation sends structured error feedback back to the LLM so it can self-correct.

### Unconstrained baseline

```bash
uv run python -m src.pipeline.orchestrator --unconstrained
```

Same prompt, same temperature, same data — no validation wrapper. This is the control condition.

### Single-country run (useful for debugging)

```bash
uv run python -m src.pipeline.orchestrator --country FRA
```

### Save output to a file

```bash
uv run python -m src.pipeline.orchestrator --output results/run_001.json
```

---

## Reproducing Results

Pre-generated results from all 20 experimental runs (5 constrained + 5 unconstrained × 2 models) are included in `results/`. To reproduce the full analysis from these results without re-running the LLM:

### 1. Load all results into the database

```bash
uv run python load_all.py
```

This reads every JSON file in `results/` and inserts them into the SQLite database at `data/travel_advisory.db`.

### 2. Run the stability analysis

```bash
uv run python analyze.py
```

This compares constrained vs. unconstrained runs for both models and prints the stability metrics (row-level Jaccard similarity with bootstrap confidence intervals and volatility reduction).

### Running your own experiments

To re-run the full experiment from scratch (requires a `NAVIGATOR_API_KEY`):

```bash
# GPT-OSS-120B runs (set OPENAI_MODEL=gpt-oss-120b in .env)
uv run python -m src.pipeline.orchestrator --output results/constrained_run_1.json
uv run python -m src.pipeline.orchestrator --unconstrained --output results/unconstrained_run_1.json

# Llama runs (change OPENAI_MODEL=llama-3.1-8b-instruct in .env)
uv run python -m src.pipeline.orchestrator --output results/constrained_run_llama_1.json
uv run python -m src.pipeline.orchestrator --unconstrained --output results/unconstrained_run_llama_1.json
```

Repeat for runs 2–5, then load and analyze as shown above.

---

## Comparing Runs

Once you have two or more runs stored, the stability analysis server computes all the metrics:

```python
from src.servers.stability_analysis import compare_runs, compare_conditions

# Compare two specific runs
result = compare_runs("run-20260326-120000-abc123", "run-20260326-140000-def456")
print(result["primary"]["row_jaccard"])  # {'mean': 0.82, 'ci_lower': 0.74, 'ci_upper': 0.87}

# Compare constrained vs. unconstrained groups (need ≥2 runs each)
result = compare_conditions(
    constrained_run_ids=["run-c1", "run-c2", "run-c3"],
    unconstrained_run_ids=["run-u1", "run-u2", "run-u3"],
)
print(result["volatility_reduction"])
```

---

## Running Tests

```bash
uv run pytest
```

141 tests across two files covering all MCP servers and the orchestrator. No API calls, no model downloads needed — the sentence-transformer encoder is mocked in tests.

```bash
uv run pytest tests/test_servers.py   # schema validation, load, stability analysis
uv run pytest tests/test_pipeline.py  # orchestrator logic, retry loop, summary stats
```

---

## How the Pipeline Works

```
State Dept RSS feed ──┐
                       ├──► LLM (temp=1.0) ──► Schema Validator ──► SQLite
News articles ────────┘         ▲ retry w/ error feedback (constrained only)
```

Five MCP servers handle the pieces:

| Server | What it does |
|---|---|
| `state_dept_extract` | Fetches travel advisories from the State Dept RSS feed and caches them |
| `news_extract` | Fetches one recent news article per country from NewsAPI and caches them |
| `schema_validation` | Validates LLM output against a strict 12-field JSON Schema (Draft 7); generates retry prompts on failure |
| `load_server` | Inserts validated records into SQLite; tracks run metadata as versioned snapshots |
| `stability_analysis` | Compares runs using row-level Jaccard similarity, entity stability, cosine similarity, and bootstrap CIs |

The **constrained** condition wraps the LLM call with `schema_validation` and retries up to 3 times with structured feedback. The **unconstrained** condition uses the exact same prompt and temperature but accepts whatever the LLM returns.

---

## Paper

The research paper is in `paper/`. It is written in LaTeX using the ACM template.

- `paper/paper.tex` — main manuscript
- `paper/citations.bib` — bibliography
- `paper/generate_figures.py` — script to regenerate all figures
- `paper/figures/` — generated PDF/PNG figures

---

## Project Structure

```
cis6930sp26-project/
├── src/
│   ├── servers/
│   │   ├── state_dept_extract.py   # MCP server: fetch & cache advisories
│   │   ├── news_extract.py         # MCP server: fetch & cache news
│   │   ├── schema_validation.py    # MCP server: validate + retry prompt
│   │   ├── load_server.py          # MCP server: SQLite persistence
│   │   └── stability_analysis.py   # MCP server: stability metrics
│   ├── pipeline/
│   │   └── orchestrator.py         # LLM orchestration + retry loop
│   └── schema/
│       ├── travel_advisory.json    # JSON Schema (Draft 7, 12 fields)
│       └── target_countries.json   # 30 target countries with ISO codes
├── tests/
│   ├── test_servers.py             # 94 tests for all MCP servers
│   └── test_pipeline.py           # 47 tests for orchestrator
├── data/
│   ├── raw/
│   │   ├── state_dept/             # Cached advisories ({CODE}.json)
│   │   └── news/                   # Cached news articles ({CODE}.json)
│   └── travel_advisory.db          # SQLite database (created on first load)
├── results/                        # Pre-generated LLM outputs (20 JSON files)
├── paper/                          # LaTeX paper + figures
│   ├── paper.tex
│   ├── citations.bib
│   ├── generate_figures.py
│   └── figures/
├── load_all.py                     # Bulk-load all results into SQLite
├── analyze.py                      # Run stability analysis across all runs
└── pyproject.toml
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NAVIGATOR_API_KEY` | *(required)* | API key for the UF HiPerGator LLM endpoint |
| `OPENAI_BASE_URL` | `https://api.ai.it.ufl.edu/v1` | LLM API base URL |
| `OPENAI_MODEL` | `gpt-oss-120b` | Model name (`gpt-oss-120b` or `llama-3.1-8b-instruct`) |
