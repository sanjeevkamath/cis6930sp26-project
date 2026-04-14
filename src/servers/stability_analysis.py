"""
StabilityAnalysisServer — MCP server for measuring LLM output volatility.

Research question: Does schema-constrained generation reduce output volatility ≥50%?

Primary metric:   row_jaccard — mean Jaccard similarity over LLM-generated token sets per country.
                  Tokens: event_type:{v} and regional_warning:{v} only.
                  risk_level/risk_label excluded (hardcoded from source, not LLM-generated).
Secondary:        reproducibility_rate, entity_stability, summary_cosine.
Research claim:   volatility_reduction_pct derived from row_jaccard means across conditions.

Missing countries (absent from either run) count as row_jaccard = 0.0 in all metrics.
The ≥50% threshold is a benchmark hypothesis; sub-50% results indicate partial stabilization.
"""

import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parents[2] / "data" / "travel_advisory.db"
SCHEMA_DIR = Path(__file__).parents[1] / "schema"

mcp = FastMCP("StabilityAnalysisServer")


# ---------------------------------------------------------------------------
# Lazy encoder — loaded once, reused across all tool calls in the process
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_expected_countries() -> list[str]:
    """Read target_countries.json and return the list of 30 alpha-3 codes."""
    path = SCHEMA_DIR / "target_countries.json"
    data = json.loads(path.read_text())
    return [entry["country_code"] for entry in data]


def _load_run_records(run_id: str) -> list[dict] | None:
    """Return all records for run_id, or None if run_id does not exist in DB."""
    if not DB_PATH.exists():
        return None
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT run_id FROM run_snapshots WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        rows = conn.execute(
            "SELECT record_json FROM records WHERE run_id = ? ORDER BY country_code",
            (run_id,),
        ).fetchall()
    return [json.loads(r["record_json"]) for r in rows]


def _load_run_snapshot(run_id: str) -> dict | None:
    """Return snapshot metadata for run_id, or None."""
    if not DB_PATH.exists():
        return None
    with _get_connection() as conn:
        row = conn.execute(
            """SELECT run_id, constrained, total, succeeded, failed,
                      schema_pass_rate, avg_attempts
               FROM run_snapshots WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def _index_records(records: list[dict]) -> dict[str, dict]:
    """Return {country_code: record} mapping."""
    return {r["country_code"]: r for r in records}


def _row_tokens(record: dict) -> frozenset:
    """Convert a record to a set of typed tokens for Jaccard comparison.

    Tokens: "event_type:{v}" (one per item), "regional_warning:{v}" (one per item).

    risk_level and risk_label are intentionally excluded: the orchestrator
    prompt passes these values from the State Department cache and instructs
    the LLM to echo them unchanged, so they are not LLM-generated outputs
    and contribute no signal to output volatility measurement.

    Free-text fields are also excluded: advisory_summary, entry_exit_requirements,
    run_id, timestamp, citations, country_name, news_headlines.
    """
    tokens = set()

    for et in record.get("event_types", []):
        tokens.add(f"event_type:{et}")

    for rw in record.get("regional_warnings", []):
        tokens.add(f"regional_warning:{rw}")

    return frozenset(tokens)


def _jaccard(set_a: frozenset, set_b: frozenset) -> float:
    """Standard Jaccard similarity. Both empty → 1.0."""
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 1.0


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity of two vectors, clipped to [−1, 1]."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0))


def _bootstrap_ci(values: list[float], n: int = 1000) -> dict:
    """Bootstrap 95% CI over a list of floats (2.5th / 97.5th percentile).

    Returns {"mean": float, "ci_lower": float, "ci_upper": float}.
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    )
    return {
        "mean": round(mean, 4),
        "ci_lower": round(float(np.percentile(boot_means, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_means, 97.5)), 4),
    }


def _compute_pair_metrics(
    records_a: list[dict],
    records_b: list[dict],
    expected_codes: list[str],
) -> dict:
    """Compute all stability metrics for one run pair.

    Missing countries (absent from either run) contribute 0.0 to every metric.

    Returns:
        per_country         : list of per-country metric dicts
        reproducibility_rate: float
        row_jaccards        : list[float] — one per expected country
        risk_level_matches  : list[float]
        risk_label_matches  : list[float]
        event_types_jaccards: list[float]
        cosine_sims         : list[float]
    """
    idx_a = _index_records(records_a)
    idx_b = _index_records(records_b)

    # Identify countries present in both runs — batch-embed their summaries
    both_codes = [c for c in expected_codes if c in idx_a and c in idx_b]
    embed_map: dict[str, tuple] = {}
    if both_codes:
        encoder = _get_encoder()
        texts_a = [idx_a[c].get("advisory_summary", "") for c in both_codes]
        texts_b = [idx_b[c].get("advisory_summary", "") for c in both_codes]
        all_vecs = encoder.encode(texts_a + texts_b, show_progress_bar=False)
        half = len(both_codes)
        for i, code in enumerate(both_codes):
            embed_map[code] = (all_vecs[i], all_vecs[half + i])

    per_country = []
    row_jaccards: list[float] = []
    risk_level_matches: list[float] = []
    risk_label_matches: list[float] = []
    event_types_jaccards: list[float] = []
    cosine_sims: list[float] = []
    reproduced_count = 0

    for code in expected_codes:
        rec_a = idx_a.get(code)
        rec_b = idx_b.get(code)

        if rec_a is None or rec_b is None:
            per_country.append({
                "country_code": code,
                "reproduced": False,
                "row_jaccard": 0.0,
                "risk_level_match": 0,
                "risk_label_match": 0,
                "event_types_jaccard": 0.0,
                "summary_cosine": 0.0,
            })
            row_jaccards.append(0.0)
            risk_level_matches.append(0.0)
            risk_label_matches.append(0.0)
            event_types_jaccards.append(0.0)
            cosine_sims.append(0.0)
            continue

        reproduced_count += 1

        # Row Jaccard
        rj = _jaccard(_row_tokens(rec_a), _row_tokens(rec_b))

        # Entity stability
        rl_match = 1 if rec_a.get("risk_level") == rec_b.get("risk_level") else 0
        rlab_match = 1 if rec_a.get("risk_label") == rec_b.get("risk_label") else 0
        et_jac = _jaccard(
            frozenset(rec_a.get("event_types", [])),
            frozenset(rec_b.get("event_types", [])),
        )

        # Cosine similarity (from batch embeddings)
        vec_a, vec_b = embed_map[code]
        cos = _cosine(vec_a, vec_b)

        per_country.append({
            "country_code": code,
            "reproduced": True,
            "row_jaccard": round(rj, 4),
            "risk_level_match": rl_match,
            "risk_label_match": rlab_match,
            "event_types_jaccard": round(et_jac, 4),
            "summary_cosine": round(cos, 4),
        })
        row_jaccards.append(rj)
        risk_level_matches.append(float(rl_match))
        risk_label_matches.append(float(rlab_match))
        event_types_jaccards.append(et_jac)
        cosine_sims.append(cos)

    return {
        "per_country": per_country,
        "reproducibility_rate": round(reproduced_count / len(expected_codes), 4) if expected_codes else 0.0,
        "row_jaccards": row_jaccards,
        "risk_level_matches": risk_level_matches,
        "risk_label_matches": risk_label_matches,
        "event_types_jaccards": event_types_jaccards,
        "cosine_sims": cosine_sims,
    }


def _snap_meta(snap: dict | None, records: list[dict]) -> dict:
    if snap is None:
        return {
            "constrained": None,
            "total": len(records),
            "succeeded": len(records),
            "schema_pass_rate": None,
            "avg_attempts": None,
        }
    return {
        "constrained": bool(snap["constrained"]),
        "total": snap["total"],
        "succeeded": snap["succeeded"],
        "schema_pass_rate": snap["schema_pass_rate"],
        "avg_attempts": snap["avg_attempts"],
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    """
    Compare two pipeline runs and compute stability metrics.

    Primary metric: row_jaccard — mean Jaccard similarity over typed token sets.
    Countries missing from either run contribute row_jaccard = 0.0.

    Args:
        run_id_a: First run identifier.
        run_id_b: Second run identifier.

    Returns:
        Stability report with per-country breakdown and aggregate bootstrap CIs.
    """
    records_a = _load_run_records(run_id_a)
    if records_a is None:
        return {"error": "run_id not found", "run_id": run_id_a}
    if len(records_a) == 0:
        return {"error": "no records for run_id", "run_id": run_id_a}

    records_b = _load_run_records(run_id_b)
    if records_b is None:
        return {"error": "run_id not found", "run_id": run_id_b}
    if len(records_b) == 0:
        return {"error": "no records for run_id", "run_id": run_id_b}

    expected_codes = _load_expected_countries()
    m = _compute_pair_metrics(records_a, records_b, expected_codes)

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "metadata": {
            "run_a": _snap_meta(_load_run_snapshot(run_id_a), records_a),
            "run_b": _snap_meta(_load_run_snapshot(run_id_b), records_b),
        },
        "reproducibility_rate": m["reproducibility_rate"],
        "primary": {
            "row_jaccard": _bootstrap_ci(m["row_jaccards"]),
        },
        "secondary": {
            "entity_stability": {
                "risk_level_agreement": _bootstrap_ci(m["risk_level_matches"]),
                "risk_label_agreement": _bootstrap_ci(m["risk_label_matches"]),
                "event_types_jaccard": _bootstrap_ci(m["event_types_jaccards"]),
            },
            "summary_cosine": _bootstrap_ci(m["cosine_sims"]),
        },
        "per_country": m["per_country"],
    }


@mcp.tool()
def compute_stability_report(run_ids: list[str]) -> dict:
    """
    Compute stability across all C(N,2) pairs within a run set.

    Bootstrap CI is computed over per-pair means (not pooled country values),
    so each pair contributes one data point to the CI estimate.

    Args:
        run_ids: List of run identifiers — need at least 2.

    Returns:
        Aggregate stability report with schema compliance summary.
    """
    if len(run_ids) < 2:
        return {"error": "need at least 2 run_ids"}

    all_records: dict[str, list[dict]] = {}
    all_snapshots: dict[str, dict | None] = {}
    for rid in run_ids:
        records = _load_run_records(rid)
        if records is None:
            return {"error": "run_id not found", "run_id": rid}
        if len(records) == 0:
            return {"error": "no records for run_id", "run_id": rid}
        all_records[rid] = records
        all_snapshots[rid] = _load_run_snapshot(rid)

    expected_codes = _load_expected_countries()
    pairs = list(itertools.combinations(run_ids, 2))

    pair_row_jaccards: list[float] = []
    pair_repro_rates: list[float] = []
    pair_risk_level: list[float] = []
    pair_risk_label: list[float] = []
    pair_event_types: list[float] = []
    pair_cosines: list[float] = []

    for rid_a, rid_b in pairs:
        m = _compute_pair_metrics(all_records[rid_a], all_records[rid_b], expected_codes)
        pair_row_jaccards.append(float(np.mean(m["row_jaccards"])))
        pair_repro_rates.append(m["reproducibility_rate"])
        pair_risk_level.append(float(np.mean(m["risk_level_matches"])))
        pair_risk_label.append(float(np.mean(m["risk_label_matches"])))
        pair_event_types.append(float(np.mean(m["event_types_jaccards"])))
        pair_cosines.append(float(np.mean(m["cosine_sims"])))

    schema_compliance = []
    for rid in run_ids:
        snap = all_snapshots[rid]
        entry: dict = {"run_id": rid}
        if snap:
            entry["constrained"] = bool(snap["constrained"])
            entry["schema_pass_rate"] = snap["schema_pass_rate"]
            entry["avg_attempts"] = snap["avg_attempts"]
        else:
            entry["constrained"] = None
            entry["schema_pass_rate"] = None
            entry["avg_attempts"] = None
        schema_compliance.append(entry)

    return {
        "run_ids": run_ids,
        "n_runs": len(run_ids),
        "n_pairs": len(pairs),
        "schema_compliance_summary": schema_compliance,
        "primary": {
            "row_jaccard": _bootstrap_ci(pair_row_jaccards),
        },
        "secondary": {
            "reproducibility_rate": _bootstrap_ci(pair_repro_rates),
            "entity_stability": {
                "risk_level_agreement": _bootstrap_ci(pair_risk_level),
                "risk_label_agreement": _bootstrap_ci(pair_risk_label),
                "event_types_jaccard": _bootstrap_ci(pair_event_types),
            },
            "summary_cosine": _bootstrap_ci(pair_cosines),
        },
    }


@mcp.tool()
def compare_conditions(
    constrained_run_ids: list[str],
    unconstrained_run_ids: list[str],
) -> dict:
    """
    Compare constrained vs. unconstrained pipeline conditions.

    Calls compute_stability_report for each group, then derives:

        volatility_reduction_pct = (constrained_mean - unconstrained_mean)
                                   / (1 - unconstrained_mean)

    Applied to row_jaccard means. ≥50% reduction meets the benchmark hypothesis;
    sub-50% indicates partial stabilization and is still interpretable.

    Args:
        constrained_run_ids:   Run IDs from the schema-constrained condition.
        unconstrained_run_ids: Run IDs from the baseline (unconstrained) condition.

    Returns:
        Stability reports for both conditions plus the volatility reduction claim.
    """
    if len(constrained_run_ids) < 2:
        return {"error": "need at least 2 run_ids", "group": "constrained"}
    if len(unconstrained_run_ids) < 2:
        return {"error": "need at least 2 run_ids", "group": "unconstrained"}

    constrained_report = compute_stability_report(constrained_run_ids)
    if "error" in constrained_report:
        return constrained_report

    unconstrained_report = compute_stability_report(unconstrained_run_ids)
    if "error" in unconstrained_report:
        return unconstrained_report

    c_mean = constrained_report["primary"]["row_jaccard"]["mean"]
    u_mean = unconstrained_report["primary"]["row_jaccard"]["mean"]

    denom = 1.0 - u_mean
    if denom <= 0:
        reduction_pct = None
        hypothesis_met = None
        note = "Unconstrained mean is 1.0 — no volatility to reduce."
    else:
        reduction_pct = round((c_mean - u_mean) / denom, 4)
        hypothesis_met = reduction_pct >= 0.50
        note = "≥50% threshold is the benchmark; values below 50% indicate partial stabilization."

    return {
        "constrained": constrained_report,
        "unconstrained": unconstrained_report,
        "volatility_reduction": {
            "row_jaccard_constrained_mean": c_mean,
            "row_jaccard_unconstrained_mean": u_mean,
            "reduction_pct": reduction_pct,
            "hypothesis_met": hypothesis_met,
            "note": note,
        },
    }


@mcp.tool()
def get_field_stability(run_ids: list[str], field: str) -> dict:
    """
    Exploratory drill-down: compute stability for a single field across runs.

    Note: This is an exploratory tool, not the primary evaluation endpoint.
    For the main research claim, use compare_conditions / compute_stability_report.

    Valid fields: risk_level, risk_label, event_types, regional_warnings, advisory_summary.
    - Categorical fields (risk_level, risk_label): exact-match score per country.
    - Set fields (event_types, regional_warnings): Jaccard similarity per country.
    - advisory_summary: cosine similarity of sentence embeddings.

    Missing countries score 0.0 in all cases.

    Args:
        run_ids: List of run identifiers — need at least 2.
        field:   Field name to analyze.

    Returns:
        Per-pair and aggregate stability for the specified field.
    """
    VALID_FIELDS = [
        "risk_level", "risk_label", "event_types", "regional_warnings", "advisory_summary"
    ]
    if field not in VALID_FIELDS:
        return {"error": "unknown field", "valid_fields": VALID_FIELDS}

    if len(run_ids) < 2:
        return {"error": "need at least 2 run_ids"}

    all_records: dict[str, list[dict]] = {}
    for rid in run_ids:
        records = _load_run_records(rid)
        if records is None:
            return {"error": "run_id not found", "run_id": rid}
        if len(records) == 0:
            return {"error": "no records for run_id", "run_id": rid}
        all_records[rid] = records

    expected_codes = _load_expected_countries()
    pairs = list(itertools.combinations(run_ids, 2))
    pair_scores: list[float] = []
    pair_details = []

    for rid_a, rid_b in pairs:
        idx_a = _index_records(all_records[rid_a])
        idx_b = _index_records(all_records[rid_b])

        scores: list[float] = []
        country_scores = []

        if field == "advisory_summary":
            both_codes = [c for c in expected_codes if c in idx_a and c in idx_b]
            embed_map: dict[str, tuple] = {}
            if both_codes:
                encoder = _get_encoder()
                texts_a = [idx_a[c].get("advisory_summary", "") for c in both_codes]
                texts_b = [idx_b[c].get("advisory_summary", "") for c in both_codes]
                all_vecs = encoder.encode(texts_a + texts_b, show_progress_bar=False)
                half = len(both_codes)
                for i, code in enumerate(both_codes):
                    embed_map[code] = (all_vecs[i], all_vecs[half + i])

            for code in expected_codes:
                if code not in idx_a or code not in idx_b:
                    scores.append(0.0)
                    country_scores.append({"country_code": code, "score": 0.0, "present": False})
                else:
                    cos = _cosine(*embed_map[code])
                    scores.append(cos)
                    country_scores.append({"country_code": code, "score": round(cos, 4), "present": True})

        elif field in ("risk_level", "risk_label"):
            for code in expected_codes:
                rec_a = idx_a.get(code)
                rec_b = idx_b.get(code)
                if rec_a is None or rec_b is None:
                    scores.append(0.0)
                    country_scores.append({"country_code": code, "score": 0.0, "present": False})
                else:
                    match = 1.0 if rec_a.get(field) == rec_b.get(field) else 0.0
                    scores.append(match)
                    country_scores.append({"country_code": code, "score": match, "present": True})

        else:  # event_types, regional_warnings
            for code in expected_codes:
                rec_a = idx_a.get(code)
                rec_b = idx_b.get(code)
                if rec_a is None or rec_b is None:
                    scores.append(0.0)
                    country_scores.append({"country_code": code, "score": 0.0, "present": False})
                else:
                    jac = _jaccard(
                        frozenset(rec_a.get(field, [])),
                        frozenset(rec_b.get(field, [])),
                    )
                    scores.append(jac)
                    country_scores.append({"country_code": code, "score": round(jac, 4), "present": True})

        pair_mean = float(np.mean(scores)) if scores else 0.0
        pair_scores.append(pair_mean)
        pair_details.append({
            "run_id_a": rid_a,
            "run_id_b": rid_b,
            "mean_score": round(pair_mean, 4),
            "per_country": country_scores,
        })

    return {
        "field": field,
        "run_ids": run_ids,
        "n_pairs": len(pairs),
        "aggregate": _bootstrap_ci(pair_scores),
        "pairs": pair_details,
    }


if __name__ == "__main__":
    mcp.run()
