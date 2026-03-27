"""
Tests for src/servers/: schema_validation, load_server, stability_analysis.

Organized into three sections:
  1. SchemaValidationServer — validate_record, validate_batch, get_schema, get_field_constraints
  2. LoadServer            — insert_record, create_snapshot, list/get helpers
  3. StabilityAnalysisServer — helper unit tests + MCP tool integration tests

DB isolation: each test that writes to SQLite receives a fresh tmp_path via
the `tmp_db` fixture, which monkeypatches DB_PATH in both load_server and
stability_analysis modules.

Encoder isolation: tests that call stability_analysis tools that trigger
sentence embeddings use the `mock_encoder` fixture, which replaces
_get_encoder() with a fast deterministic stub (no model download needed).
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.servers.load_server as ls_module
import src.servers.stability_analysis as sa_module
from src.servers.schema_validation import (
    validate_record,
    validate_batch,
    get_schema,
    get_field_constraints,
)
from src.servers.load_server import (
    insert_record,
    insert_run_results,
    create_snapshot,
    list_snapshots,
    list_run_ids,
    get_snapshot_records,
    get_record,
)
from src.servers.stability_analysis import (
    _jaccard,
    _cosine,
    _bootstrap_ci,
    _row_tokens,
    _load_expected_countries,
    _index_records,
    compare_runs,
    compute_stability_report,
    compare_conditions,
    get_field_stability,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal valid record — all required fields present and schema-conformant
VALID_RECORD = {
    "country_name": "France",
    "country_code": "FRA",
    "risk_level": 2,
    "risk_label": "Exercise Increased Caution",
    "event_types": ["crime", "terrorism"],
    "advisory_summary": "France faces ongoing terrorism threats and petty crime in tourist areas.",
    "regional_warnings": ["Paris suburbs", "Marseille"],
    "entry_exit_requirements": "No visa required for US citizens staying under 90 days.",
    "news_headlines": ["France terror alert raised"],
    "run_id": "run-test-001",
    "timestamp": "2026-01-01T00:00:00+00:00",
}

VALID_RECORD_JPN = {
    "country_name": "Japan",
    "country_code": "JPN",
    "risk_level": 1,
    "risk_label": "Exercise Normal Precautions",
    "event_types": ["natural_disaster"],
    "advisory_summary": "Japan is generally safe. Earthquakes and typhoons are possible.",
    "regional_warnings": [],
    "entry_exit_requirements": "US citizens may visit up to 90 days without a visa.",
    "news_headlines": ["Japan earthquake preparedness"],
    "run_id": "run-test-001",
    "timestamp": "2026-01-01T00:00:00+00:00",
}


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Redirect both modules to a fresh SQLite file for each test."""
    db = tmp_path / "test_travel.db"
    monkeypatch.setattr(ls_module, "DB_PATH", db)
    monkeypatch.setattr(sa_module, "DB_PATH", db)
    return db


class _MockEncoder:
    """Deterministic stub that returns L2-normalised random unit vectors."""

    def encode(self, texts, show_progress_bar=False):
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), 16)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


@pytest.fixture()
def mock_encoder(monkeypatch):
    """Replace the lazy sentence-transformers encoder with the deterministic stub."""
    enc = _MockEncoder()
    monkeypatch.setattr(sa_module, "_encoder", enc)
    return enc


# ---------------------------------------------------------------------------
# Helpers shared across load_server and stability_analysis tests
# ---------------------------------------------------------------------------

def _make_pipeline_output(run_id, constrained, records):
    """Build a minimal pipeline_output dict (matching Orchestrator.run_pipeline output)."""
    results = [{"status": "success", "record": r} for r in records]
    return {
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "constrained": constrained,
        "model": "gpt-oss-120b",
        "results": results,
        "summary": {
            "total": len(records),
            "succeeded": len(records),
            "failed": 0,
            "schema_pass_rate": 1.0,
            "avg_attempts": 1.0,
            "retry_distribution": {1: len(records), 2: 0, 3: 0},
        },
    }


def _insert_run(run_id, constrained, records):
    """Insert a full run (records + snapshot) using load_server tools."""
    for rec in records:
        insert_record({**rec, "run_id": run_id}, run_id)
    po = _make_pipeline_output(run_id, constrained, records)
    create_snapshot(po)


# ===========================================================================
# 1. SchemaValidationServer
# ===========================================================================


class TestValidateRecord:
    def test_valid_record_passes(self):
        result = validate_record(VALID_RECORD)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["retry_prompt"] is None

    def test_missing_required_field(self):
        bad = {k: v for k, v in VALID_RECORD.items() if k != "risk_level"}
        result = validate_record(bad)
        assert result["valid"] is False
        fields = [e["field"] for e in result["errors"]]
        assert any("risk_level" in f or f == "(root)" for f in fields)

    def test_invalid_risk_level_out_of_range(self):
        bad = {**VALID_RECORD, "risk_level": 5}
        result = validate_record(bad)
        assert result["valid"] is False
        assert any("risk_level" in e["field"] for e in result["errors"])

    def test_invalid_risk_level_type(self):
        bad = {**VALID_RECORD, "risk_level": "high"}
        result = validate_record(bad)
        assert result["valid"] is False

    def test_invalid_risk_label_enum(self):
        bad = {**VALID_RECORD, "risk_label": "Danger Zone"}
        result = validate_record(bad)
        assert result["valid"] is False
        assert any("risk_label" in e["field"] for e in result["errors"])

    def test_invalid_event_type_enum(self):
        bad = {**VALID_RECORD, "event_types": ["crime", "aliens"]}
        result = validate_record(bad)
        assert result["valid"] is False

    def test_empty_event_types_fails(self):
        bad = {**VALID_RECORD, "event_types": []}
        result = validate_record(bad)
        assert result["valid"] is False

    def test_invalid_country_code_pattern(self):
        bad = {**VALID_RECORD, "country_code": "fr"}
        result = validate_record(bad)
        assert result["valid"] is False

    def test_advisory_summary_too_short(self):
        bad = {**VALID_RECORD, "advisory_summary": "Short."}
        result = validate_record(bad)
        assert result["valid"] is False

    def test_additional_property_rejected(self):
        bad = {**VALID_RECORD, "extra_field": "oops"}
        result = validate_record(bad)
        assert result["valid"] is False

    def test_retry_prompt_present_on_failure(self):
        bad = {**VALID_RECORD, "risk_level": 99}
        result = validate_record(bad)
        assert result["valid"] is False
        assert isinstance(result["retry_prompt"], str)
        assert "risk_level" in result["retry_prompt"]
        assert "Fix the following" in result["retry_prompt"]

    def test_optional_citations_accepted(self):
        rec = {**VALID_RECORD, "citations": [{"url": "https://example.com", "title": "Source"}]}
        result = validate_record(rec)
        assert result["valid"] is True

    def test_risk_level_boundary_values(self):
        for level in (1, 2, 3, 4):
            labels = {
                1: "Exercise Normal Precautions",
                2: "Exercise Increased Caution",
                3: "Reconsider Travel",
                4: "Do Not Travel",
            }
            rec = {**VALID_RECORD, "risk_level": level, "risk_label": labels[level]}
            assert validate_record(rec)["valid"] is True, f"level {level} should be valid"


class TestValidateBatch:
    def test_all_valid(self):
        result = validate_batch([VALID_RECORD, VALID_RECORD_JPN])
        assert result["total"] == 2
        assert result["passed"] == 2
        assert result["failed"] == 0
        assert result["pass_rate"] == 1.0
        assert all(r["valid"] for r in result["results"])

    def test_mixed_batch(self):
        bad = {**VALID_RECORD, "risk_level": 99}
        result = validate_batch([VALID_RECORD, bad])
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["pass_rate"] == 0.5

    def test_empty_batch(self):
        result = validate_batch([])
        assert result["total"] == 0
        assert result["pass_rate"] == 0.0

    def test_results_indexed_correctly(self):
        bad = {**VALID_RECORD, "risk_level": 0}
        result = validate_batch([VALID_RECORD, bad])
        assert result["results"][0]["index"] == 0
        assert result["results"][0]["valid"] is True
        assert result["results"][1]["index"] == 1
        assert result["results"][1]["valid"] is False


class TestGetSchema:
    def test_returns_dict_with_schema_key(self):
        schema = get_schema()
        assert isinstance(schema, dict)
        assert "$schema" in schema

    def test_required_fields_present(self):
        schema = get_schema()
        required = schema.get("required", [])
        for field in ("country_code", "risk_level", "event_types", "advisory_summary"):
            assert field in required


class TestGetFieldConstraints:
    def test_returns_all_fields(self):
        constraints = get_field_constraints()
        for field in ("risk_level", "risk_label", "event_types", "country_code"):
            assert field in constraints

    def test_required_flags_correct(self):
        constraints = get_field_constraints()
        assert constraints["risk_level"]["required"] is True
        assert constraints["country_code"]["required"] is True
        # citations is optional
        assert constraints["citations"]["required"] is False

    def test_enum_values_present(self):
        constraints = get_field_constraints()
        assert "allowed_values" in constraints["risk_label"]
        assert "Exercise Normal Precautions" in constraints["risk_label"]["allowed_values"]


# ===========================================================================
# 2. LoadServer
# ===========================================================================


class TestInsertRecord:
    def test_insert_returns_inserted_true(self, tmp_db):
        result = insert_record(VALID_RECORD, "run-001")
        assert result["inserted"] is True
        assert result["country_code"] == "FRA"
        assert result["run_id"] == "run-001"

    def test_insert_missing_country_code(self, tmp_db):
        bad = {k: v for k, v in VALID_RECORD.items() if k != "country_code"}
        result = insert_record(bad, "run-001")
        assert result["inserted"] is False
        assert "error" in result

    def test_upsert_idempotent(self, tmp_db):
        insert_record(VALID_RECORD, "run-001")
        result = insert_record(VALID_RECORD, "run-001")
        assert result["inserted"] is True
        # Verify only one row exists
        conn = sqlite3.connect(tmp_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM records WHERE run_id='run-001' AND country_code='FRA'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_upsert_updates_existing(self, tmp_db):
        insert_record(VALID_RECORD, "run-001")
        updated = {**VALID_RECORD, "risk_level": 3, "risk_label": "Reconsider Travel"}
        insert_record(updated, "run-001")
        retrieved = get_record("run-001", "FRA")
        assert retrieved["risk_level"] == 3

    def test_same_country_different_runs(self, tmp_db):
        insert_record(VALID_RECORD, "run-001")
        insert_record({**VALID_RECORD, "run_id": "run-002"}, "run-002")
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM records WHERE country_code='FRA'").fetchone()[0]
        conn.close()
        assert count == 2


class TestCreateSnapshot:
    def test_creates_snapshot(self, tmp_db):
        po = _make_pipeline_output("run-snap-01", True, [VALID_RECORD])
        result = create_snapshot(po)
        assert result["created"] is True
        assert result["run_id"] == "run-snap-01"

    def test_snapshot_missing_run_id(self, tmp_db):
        result = create_snapshot({})
        assert result["created"] is False
        assert "error" in result

    def test_snapshot_stores_metadata(self, tmp_db):
        po = _make_pipeline_output("run-snap-02", True, [VALID_RECORD, VALID_RECORD_JPN])
        create_snapshot(po)
        snaps = list_snapshots()
        snap = next(s for s in snaps if s["run_id"] == "run-snap-02")
        assert snap["constrained"] == 1
        assert snap["total"] == 2
        assert snap["schema_pass_rate"] == 1.0

    def test_snapshot_replace_on_duplicate(self, tmp_db):
        po = _make_pipeline_output("run-snap-03", True, [VALID_RECORD])
        create_snapshot(po)
        # Re-insert with updated schema_pass_rate
        po2 = {**po, "summary": {**po["summary"], "schema_pass_rate": 0.5}}
        create_snapshot(po2)
        snaps = list_snapshots()
        snap = next(s for s in snaps if s["run_id"] == "run-snap-03")
        assert snap["schema_pass_rate"] == 0.5


class TestInsertRunResults:
    def test_inserts_successful_records_only(self, tmp_db):
        po = _make_pipeline_output("run-batch-01", True, [VALID_RECORD, VALID_RECORD_JPN])
        po["results"].append({"status": "failed", "record": None})
        po["summary"]["total"] = 3
        po["summary"]["failed"] = 1
        result = insert_run_results(po)
        assert result["inserted"] == 2
        assert result["skipped_failed"] == 1
        assert result["snapshot_created"] is True

    def test_records_retrievable_after_batch_insert(self, tmp_db):
        po = _make_pipeline_output("run-batch-02", False, [VALID_RECORD, VALID_RECORD_JPN])
        insert_run_results(po)
        records = get_snapshot_records("run-batch-02")
        codes = {r["country_code"] for r in records}
        assert codes == {"FRA", "JPN"}


class TestListAndGet:
    def test_list_run_ids_empty(self, tmp_db):
        assert list_run_ids() == []

    def test_list_run_ids_after_insert(self, tmp_db):
        _insert_run("run-list-01", True, [VALID_RECORD])
        _insert_run("run-list-02", False, [VALID_RECORD_JPN])
        ids = list_run_ids()
        assert "run-list-01" in ids
        assert "run-list-02" in ids

    def test_list_snapshots_returns_metadata(self, tmp_db):
        _insert_run("run-ls-01", True, [VALID_RECORD])
        snaps = list_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["run_id"] == "run-ls-01"
        assert "schema_pass_rate" in snaps[0]

    def test_get_snapshot_records_returns_records(self, tmp_db):
        _insert_run("run-gsr-01", True, [VALID_RECORD, VALID_RECORD_JPN])
        records = get_snapshot_records("run-gsr-01")
        assert len(records) == 2
        codes = {r["country_code"] for r in records}
        assert codes == {"FRA", "JPN"}

    def test_get_snapshot_records_unknown_run(self, tmp_db):
        records = get_snapshot_records("nonexistent")
        assert records == []

    def test_get_record_returns_record(self, tmp_db):
        _insert_run("run-gr-01", True, [VALID_RECORD])
        rec = get_record("run-gr-01", "FRA")
        assert rec is not None
        assert rec["country_code"] == "FRA"
        assert rec["risk_level"] == 2

    def test_get_record_unknown_returns_none(self, tmp_db):
        _insert_run("run-gr-02", True, [VALID_RECORD])
        assert get_record("run-gr-02", "JPN") is None
        assert get_record("nonexistent", "FRA") is None


# ===========================================================================
# 3. StabilityAnalysisServer — helper unit tests
# ===========================================================================


class TestJaccard:
    def test_both_empty_returns_one(self):
        assert _jaccard(frozenset(), frozenset()) == 1.0

    def test_identical_sets(self):
        s = frozenset({"a", "b", "c"})
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_partial_overlap(self):
        a = frozenset({"a", "b"})
        b = frozenset({"b", "c"})
        # intersection={b}, union={a,b,c} → 1/3
        assert abs(_jaccard(a, b) - 1 / 3) < 1e-9

    def test_one_empty(self):
        assert _jaccard(frozenset({"a"}), frozenset()) == 0.0


class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine(a, b)) < 1e-6

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0])
        assert abs(_cosine(v, -v) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        v = np.array([1.0, 0.0])
        z = np.array([0.0, 0.0])
        assert _cosine(v, z) == 0.0
        assert _cosine(z, z) == 0.0

    def test_clipped_to_range(self):
        # Numerically near-identical vectors — result should stay in [-1, 1]
        v = np.array([1.0, 1e-9])
        result = _cosine(v, v)
        assert -1.0 <= result <= 1.0


class TestBootstrapCI:
    def test_single_value(self):
        ci = _bootstrap_ci([0.75])
        assert ci["mean"] == 0.75
        # With a single value all bootstrap samples equal 0.75
        assert ci["ci_lower"] == 0.75
        assert ci["ci_upper"] == 0.75

    def test_empty_list(self):
        ci = _bootstrap_ci([])
        assert ci == {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    def test_ci_ordering(self):
        values = [float(x) / 10 for x in range(11)]  # 0.0..1.0
        ci = _bootstrap_ci(values)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_constant_list(self):
        ci = _bootstrap_ci([0.5] * 20)
        assert ci["mean"] == 0.5
        assert ci["ci_lower"] == 0.5
        assert ci["ci_upper"] == 0.5

    def test_returns_rounded_floats(self):
        ci = _bootstrap_ci([0.123456789])
        # Should be rounded to 4 decimal places
        assert ci["mean"] == round(ci["mean"], 4)


class TestRowTokens:
    def test_contains_risk_level_token(self):
        tokens = _row_tokens(VALID_RECORD)
        assert "risk_level:2" in tokens

    def test_contains_risk_label_token(self):
        tokens = _row_tokens(VALID_RECORD)
        assert "risk_label:Exercise Increased Caution" in tokens

    def test_contains_event_type_tokens(self):
        tokens = _row_tokens(VALID_RECORD)
        assert "event_type:crime" in tokens
        assert "event_type:terrorism" in tokens

    def test_contains_regional_warning_tokens(self):
        tokens = _row_tokens(VALID_RECORD)
        assert "regional_warning:Paris suburbs" in tokens
        assert "regional_warning:Marseille" in tokens

    def test_excludes_free_text_fields(self):
        tokens = _row_tokens(VALID_RECORD)
        token_str = str(tokens)
        assert "advisory_summary" not in token_str
        assert "entry_exit" not in token_str
        assert "run_id" not in token_str
        assert "timestamp" not in token_str

    def test_empty_regional_warnings(self):
        rec = {**VALID_RECORD, "regional_warnings": []}
        tokens = _row_tokens(rec)
        assert not any(t.startswith("regional_warning:") for t in tokens)

    def test_returns_frozenset(self):
        assert isinstance(_row_tokens(VALID_RECORD), frozenset)


class TestLoadExpectedCountries:
    def test_returns_30_codes(self):
        codes = _load_expected_countries()
        assert len(codes) == 30

    def test_codes_are_uppercase_alpha3(self):
        codes = _load_expected_countries()
        for code in codes:
            assert len(code) == 3
            assert code.isupper()

    def test_known_codes_present(self):
        codes = _load_expected_countries()
        for code in ("FRA", "JPN", "AFG", "AUS"):
            assert code in codes


class TestIndexRecords:
    def test_maps_by_country_code(self):
        records = [VALID_RECORD, VALID_RECORD_JPN]
        idx = _index_records(records)
        assert "FRA" in idx
        assert "JPN" in idx
        assert idx["FRA"]["country_name"] == "France"

    def test_last_wins_on_duplicate(self):
        rec1 = {**VALID_RECORD, "risk_level": 1}
        rec2 = {**VALID_RECORD, "risk_level": 3}
        idx = _index_records([rec1, rec2])
        assert idx["FRA"]["risk_level"] == 3


# ===========================================================================
# 3b. StabilityAnalysisServer — MCP tool integration tests
# ===========================================================================


class TestCompareRuns:
    def test_run_id_not_found(self, tmp_db, mock_encoder):
        err = compare_runs("ghost-run", "also-ghost")
        assert err["error"] == "run_id not found"
        assert err["run_id"] == "ghost-run"

    def test_second_run_not_found(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        err = compare_runs("run-a", "ghost-run")
        assert err["error"] == "run_id not found"

    def test_valid_comparison_structure(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD, VALID_RECORD_JPN])
        _insert_run("run-b", False, [VALID_RECORD, VALID_RECORD_JPN])
        result = compare_runs("run-a", "run-b")
        assert "primary" in result
        assert "secondary" in result
        assert "per_country" in result
        assert "reproducibility_rate" in result
        assert "metadata" in result

    def test_row_jaccard_in_range(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD, VALID_RECORD_JPN])
        _insert_run("run-b", False, [VALID_RECORD, VALID_RECORD_JPN])
        result = compare_runs("run-a", "run-b")
        rj = result["primary"]["row_jaccard"]["mean"]
        assert 0.0 <= rj <= 1.0

    def test_ci_ordering(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", False, [VALID_RECORD])
        result = compare_runs("run-a", "run-b")
        rj = result["primary"]["row_jaccard"]
        assert rj["ci_lower"] <= rj["mean"] <= rj["ci_upper"]

    def test_identical_runs_high_jaccard_for_present(self, tmp_db, mock_encoder):
        # Same record in both runs — token Jaccard for those countries = 1.0
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        result = compare_runs("run-a", "run-b")
        fra = next(c for c in result["per_country"] if c["country_code"] == "FRA")
        assert fra["row_jaccard"] == 1.0
        assert fra["reproduced"] is True
        assert fra["risk_level_match"] == 1
        assert fra["risk_label_match"] == 1

    def test_missing_countries_score_zero(self, tmp_db, mock_encoder):
        # Only FRA in run-a, only JPN in run-b — both are "missing" from the other
        fra_rec = {**VALID_RECORD, "run_id": "run-a"}
        jpn_rec = {**VALID_RECORD_JPN, "run_id": "run-b"}
        _insert_run("run-a", True, [fra_rec])
        _insert_run("run-b", False, [jpn_rec])
        result = compare_runs("run-a", "run-b")
        fra = next(c for c in result["per_country"] if c["country_code"] == "FRA")
        jpn = next(c for c in result["per_country"] if c["country_code"] == "JPN")
        assert fra["row_jaccard"] == 0.0
        assert fra["reproduced"] is False
        assert jpn["row_jaccard"] == 0.0
        assert jpn["reproduced"] is False

    def test_reproducibility_rate_correct(self, tmp_db, mock_encoder):
        # Both runs have FRA and JPN → 2/30 reproduced
        _insert_run("run-a", True, [VALID_RECORD, VALID_RECORD_JPN])
        _insert_run("run-b", True, [VALID_RECORD, VALID_RECORD_JPN])
        result = compare_runs("run-a", "run-b")
        assert abs(result["reproducibility_rate"] - 2 / 30) < 1e-4

    def test_metadata_reflects_snapshots(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", False, [VALID_RECORD])
        result = compare_runs("run-a", "run-b")
        assert result["metadata"]["run_a"]["constrained"] is True
        assert result["metadata"]["run_b"]["constrained"] is False

    def test_per_country_covers_all_30(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        result = compare_runs("run-a", "run-b")
        assert len(result["per_country"]) == 30


class TestComputeStabilityReport:
    def test_fewer_than_two_runs_error(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        err = compute_stability_report(["run-a"])
        assert err["error"] == "need at least 2 run_ids"

    def test_unknown_run_id_error(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        err = compute_stability_report(["run-a", "ghost"])
        assert err["error"] == "run_id not found"

    def test_two_runs_structure(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        report = compute_stability_report(["run-a", "run-b"])
        assert report["n_runs"] == 2
        assert report["n_pairs"] == 1
        assert "primary" in report
        assert "secondary" in report
        assert "schema_compliance_summary" in report

    def test_three_runs_produces_three_pairs(self, tmp_db, mock_encoder):
        for rid in ("run-a", "run-b", "run-c"):
            _insert_run(rid, True, [VALID_RECORD])
        report = compute_stability_report(["run-a", "run-b", "run-c"])
        assert report["n_pairs"] == 3

    def test_secondary_keys_present(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        report = compute_stability_report(["run-a", "run-b"])
        sec = report["secondary"]
        assert "reproducibility_rate" in sec
        assert "entity_stability" in sec
        assert "summary_cosine" in sec
        es = sec["entity_stability"]
        assert "risk_level_agreement" in es
        assert "risk_label_agreement" in es
        assert "event_types_jaccard" in es

    def test_schema_compliance_has_all_runs(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", False, [VALID_RECORD])
        report = compute_stability_report(["run-a", "run-b"])
        ids = {e["run_id"] for e in report["schema_compliance_summary"]}
        assert ids == {"run-a", "run-b"}


class TestCompareConditions:
    def test_constrained_group_too_small(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", False, [VALID_RECORD])
        _insert_run("run-c", False, [VALID_RECORD])
        err = compare_conditions(["run-a"], ["run-b", "run-c"])
        assert err["error"] == "need at least 2 run_ids"
        assert err["group"] == "constrained"

    def test_unconstrained_group_too_small(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        _insert_run("run-c", False, [VALID_RECORD])
        err = compare_conditions(["run-a", "run-b"], ["run-c"])
        assert err["error"] == "need at least 2 run_ids"
        assert err["group"] == "unconstrained"

    def test_structure_of_result(self, tmp_db, mock_encoder):
        for rid, con in (("c1", True), ("c2", True), ("u1", False), ("u2", False)):
            _insert_run(rid, con, [VALID_RECORD])
        result = compare_conditions(["c1", "c2"], ["u1", "u2"])
        assert "constrained" in result
        assert "unconstrained" in result
        assert "volatility_reduction" in result
        vr = result["volatility_reduction"]
        assert "row_jaccard_constrained_mean" in vr
        assert "row_jaccard_unconstrained_mean" in vr
        assert "reduction_pct" in vr
        assert "hypothesis_met" in vr
        assert "note" in vr

    def test_identical_conditions_zero_reduction(self, tmp_db, mock_encoder):
        # Same records in both conditions → same row_jaccard means → reduction = 0
        for rid in ("c1", "c2", "u1", "u2"):
            _insert_run(rid, rid.startswith("c"), [VALID_RECORD])
        result = compare_conditions(["c1", "c2"], ["u1", "u2"])
        vr = result["volatility_reduction"]
        assert vr["reduction_pct"] == 0.0
        assert vr["hypothesis_met"] is False

    def test_note_always_present(self, tmp_db, mock_encoder):
        for rid in ("c1", "c2", "u1", "u2"):
            _insert_run(rid, rid.startswith("c"), [VALID_RECORD])
        result = compare_conditions(["c1", "c2"], ["u1", "u2"])
        assert isinstance(result["volatility_reduction"]["note"], str)


class TestGetFieldStability:
    def test_unknown_field_error(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        err = get_field_stability(["run-a", "run-b"], "bogus_field")
        assert err["error"] == "unknown field"
        assert "valid_fields" in err

    def test_fewer_than_two_runs_error(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        err = get_field_stability(["run-a"], "risk_level")
        assert err["error"] == "need at least 2 run_ids"

    def test_risk_level_identical_records(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        result = get_field_stability(["run-a", "run-b"], "risk_level")
        assert result["field"] == "risk_level"
        assert result["n_pairs"] == 1
        assert "aggregate" in result
        assert "pairs" in result
        # FRA is the only present country; its match score is 1.0;
        # the other 29 are missing → 0.0 → mean = 1/30
        assert abs(result["aggregate"]["mean"] - 1 / 30) < 1e-4

    def test_event_types_jaccard_identical(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        result = get_field_stability(["run-a", "run-b"], "event_types")
        assert result["aggregate"]["mean"] >= 0.0

    def test_advisory_summary_uses_cosine(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        result = get_field_stability(["run-a", "run-b"], "advisory_summary")
        assert result["field"] == "advisory_summary"
        # Aggregate mean should be in [0, 1]
        assert 0.0 <= result["aggregate"]["mean"] <= 1.0

    def test_valid_fields_accepted(self, tmp_db, mock_encoder):
        _insert_run("run-a", True, [VALID_RECORD])
        _insert_run("run-b", True, [VALID_RECORD])
        for field in ("risk_level", "risk_label", "event_types", "regional_warnings", "advisory_summary"):
            result = get_field_stability(["run-a", "run-b"], field)
            assert "error" not in result, f"field {field!r} raised error: {result}"
