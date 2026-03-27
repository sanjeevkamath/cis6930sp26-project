"""
Tests for src/pipeline/orchestrator.py.

All OpenAI API calls are mocked — no network traffic or real credentials needed.
Cached advisory/news data is also mocked so tests are fully self-contained.

Sections:
  1. Helper functions — _parse_json_response, _build_user_prompt
  2. Orchestrator construction
  3. run_country — constrained mode (validation + retry loop)
  4. run_country — unconstrained mode (single-shot, no validation)
  5. run_country — shared edge cases (missing advisory, API exception)
  6. run_pipeline — summary statistics and output shape
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest

import src.pipeline.orchestrator as orch_module
from src.pipeline.orchestrator import (
    Orchestrator,
    _parse_json_response,
    _build_user_prompt,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

ADVISORY = {
    "country_name": "France",
    "country_code": "FRA",
    "risk_level": 2,
    "risk_label": "Exercise Increased Caution",
    "advisory_summary": "France faces terrorism risks in public spaces.",
    "advisory_url": "https://travel.state.gov/france",
    "pub_date": "2026-01-01",
}

ADVISORY_JPN = {
    "country_name": "Japan",
    "country_code": "JPN",
    "risk_level": 1,
    "risk_label": "Exercise Normal Precautions",
    "advisory_summary": "Japan is generally safe. Natural disasters are possible.",
    "advisory_url": "https://travel.state.gov/japan",
    "pub_date": "2026-01-01",
}

NEWS = {
    "article": {
        "title": "France raises terror alert",
        "description": "Security forces on high alert.",
        "content": "French authorities have raised the terror alert level.",
        "source": "Reuters",
        "published_at": "2026-01-01T12:00:00Z",
        "url": "https://reuters.com/france-alert",
    }
}

# A schema-valid record the mock LLM returns
VALID_RECORD = {
    "country_name": "France",
    "country_code": "FRA",
    "risk_level": 2,
    "risk_label": "Exercise Increased Caution",
    "event_types": ["crime", "terrorism"],
    "advisory_summary": "France faces ongoing terrorism threats and petty crime in tourist areas.",
    "regional_warnings": ["Paris suburbs"],
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

TWO_COUNTRY_LIST = [
    {"country_name": "France", "country_code": "FRA"},
    {"country_name": "Japan", "country_code": "JPN"},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


@pytest.fixture(autouse=True)
def _patch_api_key(monkeypatch):
    """Ensure NAVIGATOR_API_KEY is always set so Orchestrator.__init__ doesn't raise."""
    monkeypatch.setattr(orch_module, "API_KEY", "test-key-for-unit-tests")


@pytest.fixture()
def mock_client(monkeypatch):
    """Replace OpenAI with a mock; return the mock client instance."""
    client = MagicMock()
    monkeypatch.setattr(orch_module, "OpenAI", MagicMock(return_value=client))
    return client


@pytest.fixture()
def mock_advisory(monkeypatch):
    """Patch _load_cached_advisory to return ADVISORY for FRA, ADVISORY_JPN for JPN, None otherwise."""
    advisories = {"FRA": ADVISORY, "JPN": ADVISORY_JPN}

    def _fake_advisory(code):
        return advisories.get(code.upper())

    monkeypatch.setattr(orch_module, "_load_cached_advisory", _fake_advisory)


@pytest.fixture()
def mock_news(monkeypatch):
    """Patch _load_cached_news to return NEWS for FRA, None for everything else."""
    def _fake_news(code):
        return NEWS if code.upper() == "FRA" else None

    monkeypatch.setattr(orch_module, "_load_cached_news", _fake_news)


@pytest.fixture()
def mock_all_advisory(monkeypatch):
    """Return a generic advisory for every country (used in run_pipeline tests)."""
    def _fake_advisory(code):
        return {
            "country_name": code,
            "country_code": code.upper(),
            "risk_level": 1,
            "risk_label": "Exercise Normal Precautions",
            "advisory_summary": f"Advisory for {code}.",
            "advisory_url": "",
        }

    monkeypatch.setattr(orch_module, "_load_cached_advisory", _fake_advisory)
    monkeypatch.setattr(orch_module, "_load_cached_news", lambda code: None)


# ---------------------------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json_object(self):
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_whitespace_stripped(self):
        assert _parse_json_response('  {"key": 1}  ') == {"key": 1}

    def test_markdown_fenced_json(self):
        content = '```json\n{"key": "value"}\n```'
        assert _parse_json_response(content) == {"key": "value"}

    def test_markdown_fenced_no_lang(self):
        content = '```\n{"key": "value"}\n```'
        assert _parse_json_response(content) == {"key": "value"}

    def test_garbage_returns_none(self):
        assert _parse_json_response("not json at all!") is None

    def test_empty_string_returns_none(self):
        assert _parse_json_response("") is None

    def test_embedded_json_extracted(self):
        content = 'Here is the record: {"key": "val"} that is all.'
        result = _parse_json_response(content)
        assert result == {"key": "val"}

    def test_nested_json(self):
        obj = {"outer": {"inner": [1, 2, 3]}}
        assert _parse_json_response(json.dumps(obj)) == obj


class TestBuildUserPrompt:
    def test_contains_country_name_and_code(self):
        prompt = _build_user_prompt(ADVISORY, None, "run-001", "2026-01-01T00:00:00+00:00")
        assert "France" in prompt
        assert "FRA" in prompt

    def test_contains_run_id_and_timestamp(self):
        prompt = _build_user_prompt(ADVISORY, None, "run-xyz", "2026-03-01T00:00:00+00:00")
        assert "run-xyz" in prompt
        assert "2026-03-01T00:00:00+00:00" in prompt

    def test_contains_risk_level_and_label(self):
        prompt = _build_user_prompt(ADVISORY, None, "run-001", "2026-01-01T00:00:00+00:00")
        assert "2" in prompt
        assert "Exercise Increased Caution" in prompt

    def test_no_news_message_when_none(self):
        prompt = _build_user_prompt(ADVISORY, None, "run-001", "2026-01-01T00:00:00+00:00")
        assert "No news article" in prompt

    def test_news_title_included_when_present(self):
        prompt = _build_user_prompt(ADVISORY, NEWS, "run-001", "2026-01-01T00:00:00+00:00")
        assert "France raises terror alert" in prompt

    def test_pub_date_included_when_present(self):
        prompt = _build_user_prompt(ADVISORY, None, "run-001", "2026-01-01T00:00:00+00:00")
        assert "2026-01-01" in prompt

    def test_advisory_without_pub_date(self):
        adv = {k: v for k, v in ADVISORY.items() if k != "pub_date"}
        prompt = _build_user_prompt(adv, None, "run-001", "2026-01-01T00:00:00+00:00")
        assert "France" in prompt  # should not raise


# ---------------------------------------------------------------------------
# 2. Orchestrator construction
# ---------------------------------------------------------------------------

class TestOrchestratorConstruction:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setattr(orch_module, "API_KEY", None)
        with pytest.raises(RuntimeError, match="NAVIGATOR_API_KEY"):
            Orchestrator()

    def test_default_constrained_is_true(self, mock_client):
        orc = Orchestrator()
        assert orc.constrained is True

    def test_unconstrained_flag(self, mock_client):
        orc = Orchestrator(constrained=False)
        assert orc.constrained is False

    def test_default_temperature(self, mock_client):
        orc = Orchestrator()
        assert orc.temperature == 1.0

    def test_custom_parameters(self, mock_client):
        orc = Orchestrator(model="custom-model", temperature=0.5, max_retries=5)
        assert orc.model == "custom-model"
        assert orc.temperature == 0.5
        assert orc.max_retries == 5


# ---------------------------------------------------------------------------
# 3. run_country — constrained mode
# ---------------------------------------------------------------------------

class TestRunCountryConstrained:
    def _orc(self, mock_client):
        return Orchestrator(constrained=True, request_delay=0)

    def test_success_on_first_attempt(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "success"
        assert result["attempts"] == 1
        assert result["errors"] == []
        assert result["record"]["country_code"] == "FRA"

    def test_retry_on_invalid_json(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.side_effect = [
            _make_response("this is not JSON at all"),
            _make_response(json.dumps(VALID_RECORD)),
        ]
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "success"
        assert result["attempts"] == 2

    def test_retry_on_schema_invalid_record(self, mock_client, mock_advisory, mock_news):
        bad_record = {**VALID_RECORD, "risk_level": 99}  # fails schema
        mock_client.chat.completions.create.side_effect = [
            _make_response(json.dumps(bad_record)),
            _make_response(json.dumps(VALID_RECORD)),
        ]
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "success"
        assert result["attempts"] == 2

    def test_all_retries_exhausted_returns_failed(self, mock_client, mock_advisory, mock_news):
        bad_record = {**VALID_RECORD, "risk_level": 99}
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(bad_record)
        )
        orc = Orchestrator(constrained=True, max_retries=3, request_delay=0)
        result = orc.run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "failed"
        assert result["attempts"] == 3
        assert len(result["errors"]) > 0

    def test_all_retries_non_json_returns_failed(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response("not json")
        orc = Orchestrator(constrained=True, max_retries=3, request_delay=0)
        result = orc.run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "failed"
        assert result["attempts"] == 3

    def test_retry_loop_sends_error_feedback(self, mock_client, mock_advisory, mock_news):
        bad_record = {**VALID_RECORD, "risk_level": 99}
        mock_client.chat.completions.create.side_effect = [
            _make_response(json.dumps(bad_record)),
            _make_response(json.dumps(VALID_RECORD)),
        ]
        self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        # Second call's messages should include a user correction turn
        second_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert roles.count("user") >= 2  # original + error feedback

    def test_country_code_echoed_in_result(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["country_code"] == "FRA"


# ---------------------------------------------------------------------------
# 4. run_country — unconstrained mode
# ---------------------------------------------------------------------------

class TestRunCountryUnconstrained:
    def _orc(self, mock_client):
        return Orchestrator(constrained=False, request_delay=0)

    def test_valid_json_accepted_immediately(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "success"
        assert result["attempts"] == 1

    def test_schema_invalid_json_accepted_no_retry(self, mock_client, mock_advisory, mock_news):
        # Unconstrained: accept whatever the LLM produces
        bad_record = {**VALID_RECORD, "risk_level": 99}
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(bad_record)
        )
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "success"
        assert result["attempts"] == 1
        assert mock_client.chat.completions.create.call_count == 1

    def test_non_json_returns_failed(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response("not json")
        result = self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert result["status"] == "failed"

    def test_single_api_call_only(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        self._orc(mock_client).run_country("France", "FRA", "run-001", "2026-01-01T00:00:00+00:00")
        assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# 5. run_country — shared edge cases
# ---------------------------------------------------------------------------

class TestRunCountryEdgeCases:
    def test_missing_advisory_returns_failed_no_api_call(self, mock_client, mock_advisory, mock_news):
        # XXX is not in mock_advisory
        result = Orchestrator(constrained=True, request_delay=0).run_country(
            "Nowhere", "XXX", "run-001", "2026-01-01T00:00:00+00:00"
        )
        assert result["status"] == "failed"
        assert result["attempts"] == 0
        assert "advisory" in result["errors"][0]["message"].lower()
        mock_client.chat.completions.create.assert_not_called()

    def test_api_exception_returns_failed(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")
        result = Orchestrator(constrained=True, request_delay=0).run_country(
            "France", "FRA", "run-001", "2026-01-01T00:00:00+00:00"
        )
        assert result["status"] == "failed"
        assert "Connection refused" in result["errors"][0]["message"]

    def test_result_includes_country_name(self, mock_client, mock_advisory, mock_news):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        result = Orchestrator(constrained=True, request_delay=0).run_country(
            "France", "FRA", "run-001", "2026-01-01T00:00:00+00:00"
        )
        assert result["country_name"] == "France"


# ---------------------------------------------------------------------------
# 6. run_pipeline — summary statistics and output shape
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_output_keys_present(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-test", country_list=TWO_COUNTRY_LIST)
        for key in ("run_id", "timestamp", "constrained", "model", "temperature", "results", "summary"):
            assert key in output

    def test_custom_run_id_used(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="my-custom-run", country_list=TWO_COUNTRY_LIST)
        assert output["run_id"] == "my-custom-run"

    def test_auto_generated_run_id_format(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(country_list=TWO_COUNTRY_LIST)
        assert output["run_id"].startswith("run-")

    def test_constrained_flag_propagated(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        output = Orchestrator(constrained=False, request_delay=0).run_pipeline(
            country_list=TWO_COUNTRY_LIST
        )
        assert output["constrained"] is False

    def test_results_one_per_country(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        assert len(output["results"]) == 2

    def test_summary_keys_present(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        summary = output["summary"]
        for key in ("total", "succeeded", "failed", "schema_pass_rate", "avg_attempts", "retry_distribution"):
            assert key in summary

    def test_summary_counts_correct(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        summary = output["summary"]
        assert summary["total"] == 2
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

    def test_schema_pass_rate_all_first_attempt(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        assert output["summary"]["schema_pass_rate"] == 1.0

    def test_schema_pass_rate_with_retries(self, mock_client, mock_all_advisory):
        # FRA takes 2 attempts (bad then good), JPN takes 1 → pass_rate = 1/2
        bad = {**VALID_RECORD, "risk_level": 99}
        mock_client.chat.completions.create.side_effect = [
            _make_response(json.dumps(bad)),         # FRA attempt 1 — invalid
            _make_response(json.dumps(VALID_RECORD)), # FRA attempt 2 — valid
            _make_response(json.dumps(VALID_RECORD)), # JPN attempt 1 — valid
        ]
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        assert output["summary"]["schema_pass_rate"] == 0.5

    def test_retry_distribution_keys(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        dist = output["summary"]["retry_distribution"]
        assert 1 in dist and 2 in dist and 3 in dist

    def test_avg_attempts_all_first(self, mock_client, mock_all_advisory):
        mock_client.chat.completions.create.return_value = _make_response(
            json.dumps(VALID_RECORD)
        )
        orc = Orchestrator(constrained=True, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        assert output["summary"]["avg_attempts"] == 1.0

    def test_failed_country_counted_in_summary(self, mock_client, mock_all_advisory):
        # Make FRA fail all retries, JPN succeed
        bad = {**VALID_RECORD, "risk_level": 99}
        mock_client.chat.completions.create.side_effect = [
            _make_response(json.dumps(bad)),         # FRA attempt 1
            _make_response(json.dumps(bad)),         # FRA attempt 2
            _make_response(json.dumps(bad)),         # FRA attempt 3
            _make_response(json.dumps(VALID_RECORD)), # JPN attempt 1
        ]
        orc = Orchestrator(constrained=True, max_retries=3, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        assert output["summary"]["failed"] == 1
        assert output["summary"]["succeeded"] == 1

    def test_unconstrained_pipeline_no_retries(self, mock_client, mock_all_advisory):
        # Even with schema-invalid output, unconstrained always succeeds on first call
        bad = {**VALID_RECORD, "risk_level": 99}
        mock_client.chat.completions.create.return_value = _make_response(json.dumps(bad))
        orc = Orchestrator(constrained=False, request_delay=0)
        output = orc.run_pipeline(run_id="run-x", country_list=TWO_COUNTRY_LIST)
        assert output["summary"]["succeeded"] == 2
        assert mock_client.chat.completions.create.call_count == 2
