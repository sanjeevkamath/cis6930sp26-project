# Design Review
**CIS6930 — Spring 2026**
**Sanjeev Kamath**

---

## 1. Refined Research Question

**Does schema-constrained validation with retry loops reduce the output volatility of an LLM-orchestrated multi-source integration pipeline by at least 50% compared to unconstrained generation, measured by row-level Jaccard similarity across repeated runs on U.S. travel advisory data for 30 countries?**

The research question is unchanged from the proposal. This project uses `gpt-oss-120b` 

---

## 2. System Architecture

### Component Diagram

![Component Diagram](/images/component_diagram.png)

```
┌──────────────────────────────────────────────────┐
│              StabilityAnalysis                   │
│              MCP Server                          │
│  Compares DB snapshots across runs               │
│  Computes Jaccard similarity, sensitivity score  │
└──────────────────────────────────────────────────┘
```

### Component Responsibilities and Interfaces

| Server | Responsibility | Key MCP Tools |
|---|---|---|
| **StateDeptExtractServer** | Fetch travel advisories from State Dept RSS feed. Fully deterministic. | `list_advisories()`, `fetch_advisory(country_name, country_code)`, `fetch_target_countries(country_list)`, `list_cached_countries()` |
| **NewsExtractServer** | Fetch 1 recent news article per country from NewsAPI. Budget-aware (≤30 of 100 daily calls). | `fetch_news(country_name, country_code)`, `fetch_news_for_all(country_list)`, `get_cached_news(country_code)` |
| **SchemaValidationServer** | Validate every LLM-generated row against the strict JSON Schema. Return structured error feedback for retry loop. | `validate_record(record)`, `validate_batch(records)`, `get_schema()`, `get_field_constraints()` |
| **LoadServer** | Insert validated rows into SQLite. Create versioned snapshots tagged with run ID and timestamp. | `insert_record(record, run_id)`, `create_snapshot(run_id)`, `list_snapshots()` |
| **StabilityAnalysisServer** | Compare snapshots across runs. Compute Jaccard similarity, schema compliance rate, sensitivity scores. | `compare_snapshots(run_id_a, run_id_b)`, `compute_stability_metrics(run_ids)` |

> **Change from proposal:** The proposal listed an `EntityNormalizationServer` as a sixth MCP server. This was simplified to a static `src/schema/target_countries.json` mapping file (country name → ISO 3166-1 alpha-3 code), since entity normalization requires no network calls or LLM involvement and a server would add unnecessary overhead.

### Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| MCP framework | `mcp[cli]` (FastMCP) | Native Python MCP SDK; minimal boilerplate |
| LLM | `gpt-oss-120b` via OpenAI SDK | University-hosted; reproducible across runs; OpenAI-compatible API |
| Schema validation | `jsonschema` (Draft 7) | Standard, strict; `additionalProperties: false` rejects hallucinated fields |
| Storage | SQLite (stdlib) | Zero-dependency; sufficient for 30 countries × N runs |
| HTML parsing | `beautifulsoup4` + `lxml` | Robust parsing for advisory detail pages |
| Semantic similarity | `sentence-transformers` (`all-MiniLM-L6-v2`) | Lightweight; used for sensitivity score in stability analysis |

---

## 3. Data Pipeline

### Full Data Flow

![Full Data Flow](/images/full_data_flow.png)
### Output Schema

Each pipeline run produces one row per country conforming to this schema (enforced by `SchemaValidationServer`):

| Field | Type | Constraints |
|---|---|---|
| `country_name` | string | Required |
| `country_code` | string | Required; pattern `^[A-Z]{3}$` |
| `risk_level` | integer | Required; 1–4 |
| `risk_label` | string | Required; enum of 4 State Dept labels |
| `event_types` | array[string] | Required; items from fixed 8-value taxonomy |
| `advisory_summary` | string | Required; minLength 20 |
| `regional_warnings` | array[string] | Required |
| `entry_exit_requirements` | string | Required |
| `news_headlines` | array[string] | Required |
| `citations` | array[{url, title}] | Optional |
| `run_id` | string | Required |
| `timestamp` | string | Required; ISO 8601 |

`additionalProperties: false` rejects any fields the LLM invents beyond this list.

### Data Quality and Edge Cases

| Issue | Handling |
|---|---|
| LLM hallucinates fields | `additionalProperties: false` in schema; row rejected and re-prompted |
| LLM produces out-of-range `risk_level` | Schema `minimum: 1, maximum: 4`; caught on first validation pass |
| LLM uses non-canonical country name | `country_code` pattern + ISO mapping in `target_countries.json` enforces canonical form |
| Network unavailability | All raw data cached locally; pipeline runs fully offline after initial fetch |
| NewsAPI budget | Server skips cached countries; 30 countries × 1 article = 30 of 100 daily calls |
| 3 consecutive validation failures | Row flagged `status: failed` in DB; counted in schema compliance metrics |

---

## 4. Implementation Plan

### Current Status (as of March 4, 2026)

**Completed:**
- `StateDeptExtractServer` — fully implemented and tested
- `NewsExtractServer` — fully implemented and tested
- `SchemaValidationServer` — fully implemented and tested
- All 30 country advisories cached in `data/raw/state_dept/`
- All 30 news articles cached in `data/raw/news/`
- JSON Schema defined with all constraints
- `pyproject.toml` with all dependencies; `uv sync` installs in one step

**Remaining:**
- `LoadServer` (SQLite insert + snapshot versioning)
- `StabilityAnalysisServer` (Jaccard, cosine similarity metrics)
- LLM Orchestrator (prompt construction, retry loop)
- Deterministic ETL baseline
- Evaluation experiments (3 reproducibility runs + 5 perturbation runs)

### Weekly Milestones

| Week | Dates | Tasks |
|---|---|---|
| **3** | Mar 3–9 | `LoadServer`: SQLite schema, insert, snapshot versioning. `StabilityAnalysisServer`: Jaccard similarity, field-level diff. LLM Orchestrator skeleton: prompt construction, single-country run. |
| **4** | Mar 10–16 | LLM retry loop (up to 3 retries). Full end-to-end pipeline for all 30 countries. Deterministic ETL baseline. First reproducibility run (run-001). |
| **5** | Mar 17–23 | 2 more reproducibility runs (run-002, run-003). Schema compliance metrics. Begin perturbation experiments (swap 1 article per country). Code Checkpoint submission. |
| **6** | Mar 24–30 | 5 perturbation runs. 10-country ground truth annotation. Compute stability, accuracy, cost metrics. Draft Paper. |
| **7** | Mar 31–Apr 6 | Full comparative analysis (constrained LLM vs. deterministic ETL). Single-shot LLM baseline if time permits. Refine paper. |
| **8** | Apr 7–13 | Polish final paper. Architecture diagrams for submission. Final Paper due. |
| **9** | Apr 20 | Presentation (10 min + Q&A). |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HiPerGator model unavailable or rate-limited | Medium | High | Cache all LLM outputs per run; retry with exponential backoff; document any downtime |
| LLM never achieves <3 retries per row | Low | Medium | Schema prompt engineering; `get_field_constraints()` injects concise constraint summary into every prompt |
| NewsAPI results irrelevant to safety | Low | Low | By design — irrelevant articles test pipeline robustness; stability metrics measure LLM response to noise |
| State Dept advisory levels change mid-study | Low | Medium | All raw data cached at fetch time; runs use cached data for reproducibility |
| Annotation disagreement on ground truth | Medium | Medium | Two annotators; Cohen's kappa measured; disagreements resolved by discussion before metrics computed |
| Perturbation causes row splits (one country → two rows) | Low | Medium | LoadServer normalizes by `country_code`; split rows flagged for manual review |

---

## 5. Preliminary Results

### Working Components

All three extraction and validation servers are implemented and passing tests against live data.

**StateDeptExtractServer** — fetches all 213+ country advisories from the State Dept RSS feed in a single HTTP request. Parsed and cached all 30 target countries:

```
Fetched: 30 / 30    Failed: 0
Risk distribution: {Level 1: 6, Level 2: 12, Level 3: 6, Level 4: 6}

L1  AUS  Australia              ['health']
L1  CAN  Canada                 ['other']
L1  JPN  Japan                  ['other']
L1  NZL  New Zealand            ['health']
L1  CHE  Switzerland            ['health']
L1  NOR  Norway                 ['other']
L2  FRA  France                 ['terrorism', 'civil_unrest', 'conflict', 'crime']
L2  DEU  Germany                ['terrorism', 'conflict']
L2  BRA  Brazil                 ['kidnapping', 'conflict', 'crime', 'health']
L2  CHN  China                  ['terrorism', 'civil_unrest', 'conflict']
L2  IND  India                  ['terrorism', 'conflict', 'crime']
L2  IDN  Indonesia              ['kidnapping', 'terrorism', 'civil_unrest', ...]
L2  KEN  Kenya                  ['kidnapping', 'terrorism', 'civil_unrest', ...]
L2  MEX  Mexico                 ['kidnapping', 'terrorism', 'crime']
L2  ZAF  South Africa           ['kidnapping', 'terrorism', 'civil_unrest', ...]
L2  THA  Thailand               ['civil_unrest', 'conflict', 'health']
L2  TUR  Turkey                 ['terrorism', 'civil_unrest', 'conflict', 'health']
L2  EGY  Egypt                  ['terrorism', 'civil_unrest', 'conflict', ...]
L3  NPL  Nepal                  ['civil_unrest', 'conflict', 'health']
L3  COL  Colombia               ['kidnapping', 'terrorism', 'civil_unrest', ...]
L3  ETH  Ethiopia               ['kidnapping', 'terrorism', 'civil_unrest', ...]
L3  PAK  Pakistan               ['kidnapping', 'terrorism', 'civil_unrest', ...]
L3  NGA  Nigeria                ['kidnapping', 'terrorism', 'civil_unrest', ...]
L3  PNG  Papua New Guinea       ['kidnapping', 'civil_unrest', 'conflict', ...]
L4  AFG  Afghanistan            ['kidnapping', 'terrorism', 'civil_unrest', ...]
L4  PRK  North Korea            ['other']
L4  RUS  Russia                 ['terrorism', 'civil_unrest', 'conflict']
L4  SYR  Syria                  ['kidnapping', 'terrorism', 'conflict', ...]
L4  YEM  Yemen                  ['kidnapping', 'terrorism', 'conflict', ...]
L4  MMR  Burma (Myanmar)        ['civil_unrest', 'conflict', 'crime', 'health']
```

**NewsExtractServer** — fetched 1 article per country (30 of 100 daily API calls). All results cached; future runs cost zero additional API calls.

**SchemaValidationServer** — validated against a 12-field strict schema. Confirmed correct behavior on both valid records and intentionally broken inputs:

| Test | Result |
|---|---|
| Valid record (France, Level 2) | `valid: true`, 0 errors |
| Invalid record (7 wrong fields, 4 missing, 1 hallucinated) | `valid: false`, 11 errors caught |
| Retry prompt generated | Correctly lists all 11 issues with field paths |
| Hallucinated field rejected | `additionalProperties: false` catches `hallucinated_field` |
| Out-of-taxonomy `event_type` | `'volcano'` rejected; allowed values listed in error |

Sample retry prompt output (demonstrating the feedback loop):
```
The previous output failed schema validation. Fix the following issues:
  1. Field 'country_code': 'fr' does not match '^[A-Z]{3}$'
  2. Field 'event_types.0': 'volcano' is not one of ['crime', 'terrorism', ...]
  3. Field 'risk_label': 'Pretty Safe' is not one of [...]
  4. Field 'risk_level': 7 is greater than the maximum of 4
  5–11. [missing required fields, hallucinated field]
Return ONLY valid JSON matching the schema. Do not include any explanation or markdown.
```

### Feasibility Assessment

The three completed servers demonstrate that:
1. All data sources are accessible and parseable without API key issues
2. The schema constraint mechanism works as designed — errors are caught and actionable feedback is generated
3. The full raw dataset (30 advisories + 30 news articles) is cached and pipeline-ready
4. The `gpt-oss-120b` endpoint and NewsAPI keys are confirmed and configured

The remaining work (LoadServer, StabilityAnalysisServer, LLM Orchestrator) follows established patterns from the completed servers and presents no novel technical risk.
