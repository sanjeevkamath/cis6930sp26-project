# Preliminary Results

The full stability experiment (5 constrained + 5 unconstrained runs across all 30 countries)
hasn't been run yet — that's the next phase. But all the infrastructure is in place and a
pilot run through the real LLM confirms the pipeline works end-to-end. This document
summarizes what's been collected and verified so far.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Target countries | 30 |
| Data sources | 2 (U.S. State Dept RSS feed, NewsAPI) |
| State Dept advisories cached | 30 / 30 |
| News articles cached | 30 / 30 |
| Advisory data date | March 2026 (fetched once, frozen for reproducibility) |
| Schema fields | 12 (strict Draft-7, `additionalProperties: false`) |

The dataset spans a deliberate range of risk profiles to stress-test stability at different
advisory levels:

| Risk Level | Label | Countries |
|-----------|-------|-----------|
| 1 | Exercise Normal Precautions | 6 (AUS, CAN, JPN, NZL, CHE, NOR) |
| 2 | Exercise Increased Caution | 12 (FRA, DEU, BRA, CHN, IND, IDN, KEN, MEX, ZAF, THA, TUR, EGY) |
| 3 | Reconsider Travel | 6 (NPL, COL, ETH, PAK, NGA, PNG) |
| 4 | Do Not Travel | 6 (AFG, PRK, RUS, SYR, YEM, MMR) |

---

## Extraction Results

### State Department Advisories

| Field | Completeness |
|-------|-------------|
| `country_name` | 30 / 30 (100%) |
| `country_code` | 30 / 30 (100%) |
| `risk_level` | 30 / 30 (100%) |
| `risk_label` | 30 / 30 (100%) |
| `event_types` | 30 / 30 (100%) |
| `advisory_summary` | 30 / 30 (100%) |
| `advisory_url` | 30 / 30 (100%) |
| `pub_date` | 30 / 30 (100%) |

Event type distribution across the 30 cached advisories (non-exclusive — most countries have
multiple types):

| Event Type | Countries |
|------------|-----------|
| `conflict` | 21 |
| `terrorism` | 18 |
| `civil_unrest` | 18 |
| `health` | 17 |
| `crime` | 16 |
| `kidnapping` | 13 |
| `other` | 4 |
| `natural_disaster` | 3 |

### News Articles

| Field | Completeness |
|-------|-------------|
| `title` | 30 / 30 (100%) |
| `description` | 30 / 30 (100%) |
| `content` | 30 / 30 (100%) |
| `source` | 30 / 30 (100%) |
| `url` | 30 / 30 (100%) |
| `published_at` | 30 / 30 (100%) |

Both datasets are fully cached. All subsequent pipeline runs read from local files — no
network calls, no API budget consumed, and data is frozen so results are reproducible.

---

## Pipeline Mechanics — Pilot Run

A two-country pilot run through the live `gpt-oss-120b` endpoint confirmed end-to-end
functionality:

| Country | Risk Level | Event Types (LLM output) | Attempts | Status |
|---------|-----------|--------------------------|----------|--------|
| France (FRA) | 2 — Exercise Increased Caution | terrorism, civil_unrest, crime | 1 | success |
| Japan (JPN) | 1 — Exercise Normal Precautions | other | 1 | success |

Both countries passed schema validation on the first attempt (`schema_pass_rate = 1.0`,
`avg_attempts = 1.0`). The constrained and unconstrained conditions were both tested and
both records were stored in the SQLite database under separate run IDs.

### Schema Validation — Stress Test

The validator was tested against a deliberately broken record (7 wrong field values,
4 missing required fields, 1 hallucinated field) to confirm it catches everything
the schema defines:

| Failure injected | Caught? |
|------------------|---------|
| `risk_level: 7` (max is 4) | ✓ |
| `risk_label: "Pretty Safe"` (not in enum) | ✓ |
| `event_types: ["volcano"]` (not in taxonomy) | ✓ |
| `country_code: "fr"` (not uppercase alpha-3) | ✓ |
| `advisory_summary: "ok"` (under 20 chars) | ✓ |
| 4 missing required fields | ✓ |
| `hallucinated_field: "..."` (not in schema) | ✓ |

All 11 errors were caught in a single validation pass, and the retry prompt correctly
named every offending field. The `additionalProperties: false` constraint is doing real
work — it's the main line of defense against the LLM inventing fields that aren't in
the schema.

---

## Test Coverage

| File | Tests | Status |
|------|-------|--------|
| `tests/test_servers.py` | 94 | ✓ all pass |
| `tests/test_pipeline.py` | 47 | ✓ all pass |
| **Total** | **141** | |

Tests cover schema validation edge cases, SQLite upsert idempotency, stability metric
helper functions (`_jaccard`, `_cosine`, `_bootstrap_ci`), retry loop behavior, and all
MCP tool error paths. The orchestrator tests mock the OpenAI client, so they run
without any API calls.

---

## Observations

- **Data quality is not the variable.** Both data sources return 100% complete fields,
  which is what we want — the study is about LLM output consistency, not data collection
  reliability. With noisy input data we'd have a confound.

- **The retry loop has real teeth.** In constrained mode, the LLM gets structured feedback
  listing every field path and error message. In early manual tests, the model consistently
  corrected schema violations on the second attempt. The pilot run didn't need retries, but
  that's expected — the prompt is well-engineered and the model is capable.

- **Risk level 4 countries may be interesting.** AFG, SYR, YEM, MMR, PRK, RUS are all
  Do Not Travel. The LLM has less structured advisory text to work from for some of these,
  and the news articles tend to be more volatile. These countries may drive higher variance
  in the stability metrics.

- **`other` and `natural_disaster` are rare in the cache.** Only 3 countries have
  `natural_disaster` in their cached event types (JPN being one). The LLM may add or drop
  these categories inconsistently across runs, which will show up in `event_types_jaccard`.

---

## Next Steps

- [ ] Run 5 constrained pipeline runs across all 30 countries, storing each in the DB
- [ ] Run 5 unconstrained baseline runs under identical conditions
- [ ] Call `compare_conditions()` to compute the volatility reduction across both groups
- [ ] Report bootstrap 95% CIs for `row_jaccard`, `entity_stability`, and `summary_cosine`
- [ ] Investigate whether Level 4 countries show higher variance than Level 1, as expected
- [ ] Write up full results in the draft paper