"""
LLM Orchestrator — synthesizes State Dept + news data into validated travel risk records.

Architecture
------------
  StateDeptExtractServer (cached) ─┐
                                    ├─► LLM (gpt-oss-120b) ─► SchemaValidationServer ─► result
  NewsExtractServer (cached)      ─┘         ▲ retry (up to MAX_RETRIES times)

Design decisions
----------------
SYSTEM_PROMPT
    Frozen module-level constant. Never modified between runs or experimental
    conditions. Prompt design is held constant to prevent it from confounding
    stability measurements. The only difference between the constrained and
    unconstrained conditions is whether the retry loop is applied.

Temperature
    1.0 (OpenAI default)

Model / API
    gpt-oss-120b via UF HiPerGator (OPENAI_BASE_URL env var).
    API key from NAVIGATOR_API_KEY. Uses the OpenAI Python SDK with a
    custom base_url (the endpoint is OpenAI-compatible).

Schema constraint mechanism
    NOT using OpenAI structured outputs / JSON mode — university endpoint
    compatibility is not guaranteed. Constraints are enforced via:
      1. Explicit rules in SYSTEM_PROMPT
      2. SchemaValidationServer.validate_record() after every LLM call
      3. Retry loop (up to MAX_RETRIES=3) with structured error feedback
    This retry loop IS the architectural intervention under study.

Direct imports
    validate_record() is imported directly from the SchemaValidationServer
    module rather than running it as a separate MCP process. All raw data is
    read from local JSON cache files. This avoids process-spawn overhead for
    single-machine pipeline runs.

constrained parameter
    Orchestrator(constrained=True)  → applies validation + retry loop
    Orchestrator(constrained=False) → single-shot, no validation (baseline)
    Same prompt, same temperature, same data — only the wrapper differs.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Direct import — validate_record is still callable after @mcp.tool() decoration
from src.servers.schema_validation import validate_record

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.ai.it.ufl.edu/v1")
API_KEY = os.getenv("NAVIGATOR_API_KEY")
TEMPERATURE = 1.0   # Intentional — measure natural variance
MAX_RETRIES = 3

STATE_DEPT_CACHE = Path(__file__).parents[2] / "data" / "raw" / "state_dept"
NEWS_CACHE = Path(__file__).parents[2] / "data" / "raw" / "news"
SCHEMA_DIR = Path(__file__).parents[1] / "schema"

# ---------------------------------------------------------------------------
# FROZEN SYSTEM PROMPT
#
# This constant must not be modified between experimental runs or conditions.
# It is identical for both the constrained and unconstrained conditions.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a travel risk analyst integrating data from the U.S. State Department and recent news articles.

Given structured advisory data and a news article for a country, produce a single JSON object that synthesizes both sources into a unified travel risk assessment.

STRICT RULES — violations will cause your output to be rejected:
- Output ONLY valid JSON. No markdown, no code fences, no explanation before or after.
- All required fields must be present: country_name, country_code, risk_level, risk_label, event_types, advisory_summary, regional_warnings, entry_exit_requirements, news_headlines, run_id, timestamp.
- risk_level: integer 1–4 only (from the State Department advisory — do not change it).
- risk_label: must be exactly one of: "Exercise Normal Precautions", "Exercise Increased Caution", "Reconsider Travel", "Do Not Travel" (match the advisory exactly).
- event_types: non-empty array; each item must be exactly one of: crime, terrorism, civil_unrest, health, natural_disaster, kidnapping, conflict, other.
- advisory_summary: 2–4 sentence synthesis of the advisory text (minimum 20 characters).
- regional_warnings: array of strings naming specific high-risk regions mentioned (use [] if none).
- entry_exit_requirements: string summarizing visa/passport/entry requirements (use "" if not mentioned).
- news_headlines: array containing the news article title(s) used as input.
- country_code: exactly 3 uppercase letters (ISO 3166-1 alpha-3).
- citations: optional array of {"url": "...", "title": "..."} objects for sources used.
- Do not add any fields not listed above.
- Include run_id and timestamp exactly as provided to you — do not alter them."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_cached_advisory(country_code: str) -> dict | None:
    path = STATE_DEPT_CACHE / f"{country_code.upper()}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _load_cached_news(country_code: str) -> dict | None:
    path = NEWS_CACHE / f"{country_code.upper()}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _build_user_prompt(
    advisory: dict,
    news: dict | None,
    run_id: str,
    timestamp: str,
) -> str:
    """Build the per-country user prompt. Structure is deterministic — only data varies."""
    article = (news or {}).get("article", {})

    lines = [
        "Generate a travel risk assessment JSON record for the following country.",
        "",
        "=== STATE DEPARTMENT ADVISORY ===",
        f"Country: {advisory['country_name']} ({advisory['country_code']})",
        f"Risk Level: {advisory['risk_level']} — {advisory['risk_label']}",
        f"Advisory Text: {advisory.get('advisory_summary', '')}",
        f"Advisory URL: {advisory.get('advisory_url', '')}",
    ]

    if advisory.get("pub_date"):
        lines.append(f"Published: {advisory['pub_date']}")

    lines += ["", "=== NEWS ARTICLE ==="]
    if article:
        lines += [
            f"Title: {article.get('title', 'N/A')}",
            f"Description: {article.get('description', 'N/A')}",
            f"Content excerpt: {article.get('content', 'N/A')}",
            f"Source: {article.get('source', 'N/A')}",
            f"Published: {article.get('published_at', 'N/A')}",
            f"URL: {article.get('url', 'N/A')}",
        ]
    else:
        lines.append("No news article available for this country.")

    lines += [
        "",
        "=== INCLUDE THESE EXACT VALUES UNCHANGED ===",
        f'run_id: "{run_id}"',
        f'timestamp: "{timestamp}"',
        f'country_name: "{advisory["country_name"]}"',
        f'country_code: "{advisory["country_code"]}"',
        f'risk_level: {advisory["risk_level"]}',
        f'risk_label: "{advisory["risk_label"]}"',
    ]

    return "\n".join(lines)


def _parse_json_response(content: str) -> dict | None:
    """Extract a JSON object from the LLM response, stripping common wrapping."""
    content = content.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        inner_lines = [
            line for line in content.split("\n")
            if not line.strip().startswith("```")
        ]
        content = "\n".join(inner_lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Last resort: find the outermost {...} block
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordinates the LLM synthesis step of the pipeline.

    Parameters
    ----------
    constrained : bool
        True  → apply SchemaValidationServer validation + retry loop (experimental condition)
        False → single-shot generation, no validation (unconstrained baseline)
    model, temperature, max_retries
        Identical across both conditions to isolate the effect of the constraint.
    """

    def __init__(
        self,
        model: str = MODEL,
        temperature: float = TEMPERATURE,
        max_retries: int = MAX_RETRIES,
        constrained: bool = True,
        request_delay: float = 1.0,
    ):
        if not API_KEY:
            raise RuntimeError("NAVIGATOR_API_KEY not set in environment")

        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.constrained = constrained
        self.request_delay = request_delay  # seconds between API calls (rate limiting)

    # ------------------------------------------------------------------
    # Single-country processing
    # ------------------------------------------------------------------

    def run_country(
        self,
        country_name: str,
        country_code: str,
        run_id: str,
        timestamp: str,
    ) -> dict:
        """
        Produce one validated travel risk record for a single country.

        Returns a result dict with:
            status       : "success" | "failed"
            record       : the validated JSON record (if success)
            attempts     : number of LLM calls made (1–MAX_RETRIES)
            errors       : list of validation error dicts from the final failed attempt
            country_code : echoed for easy indexing
        """
        advisory = _load_cached_advisory(country_code)
        if advisory is None:
            return {
                "status": "failed",
                "country_code": country_code,
                "country_name": country_name,
                "record": None,
                "attempts": 0,
                "errors": [{"field": "(root)", "message": "No cached advisory found"}],
            }

        news = _load_cached_news(country_code)
        user_prompt = _build_user_prompt(advisory, news, run_id, timestamp)

        # Conversation history — retry loop appends to this
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_errors: list[dict] = []
        attempts = 0

        max_attempts = self.max_retries if self.constrained else 1

        for attempt in range(1, max_attempts + 1):
            attempts = attempt

            # --- LLM call ---
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
            except Exception as exc:
                return {
                    "status": "failed",
                    "country_code": country_code,
                    "country_name": country_name,
                    "record": None,
                    "attempts": attempts,
                    "errors": [{"field": "(api)", "message": str(exc)}],
                }

            raw_content = response.choices[0].message.content or ""

            # --- Parse JSON ---
            record = _parse_json_response(raw_content)
            if record is None:
                last_errors = [{"field": "(json)", "message": "Response is not valid JSON", "invalid_value": repr(raw_content[:200])}]
                if not self.constrained:
                    break
                # Add the failed assistant turn and a correction request
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": "Your response was not valid JSON. Return ONLY a valid JSON object with no surrounding text or markdown.",
                })
                continue

            # --- Schema validation (constrained mode only) ---
            if self.constrained:
                validation = validate_record(record)
                if validation["valid"]:
                    return {
                        "status": "success",
                        "country_code": country_code,
                        "country_name": country_name,
                        "record": record,
                        "attempts": attempts,
                        "errors": [],
                    }

                last_errors = validation["errors"]

                if attempt < max_attempts:
                    # Append this exchange to the conversation and retry
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user",
                        "content": validation["retry_prompt"],
                    })
            else:
                # Unconstrained: accept whatever the LLM produced
                return {
                    "status": "success",
                    "country_code": country_code,
                    "country_name": country_name,
                    "record": record,
                    "attempts": attempts,
                    "errors": [],
                }

        # All attempts exhausted without a valid record
        return {
            "status": "failed",
            "country_code": country_code,
            "country_name": country_name,
            "record": None,
            "attempts": attempts,
            "errors": last_errors,
        }

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        run_id: str | None = None,
        country_list: list[dict] | None = None,
    ) -> dict:
        """
        Run the full pipeline for all target countries.

        Parameters
        ----------
        run_id
            Unique identifier for this run. Auto-generated if not provided.
        country_list
            List of {country_name, country_code} dicts.
            Defaults to all 30 countries in target_countries.json.

        Returns
        -------
        {
            run_id      : str,
            timestamp   : str,
            constrained : bool,
            model       : str,
            temperature : float,
            results     : list[dict],   # one per country
            summary     : {
                total, succeeded, failed,
                schema_pass_rate,          # fraction that passed on first attempt
                avg_attempts,              # mean LLM calls per country
                retry_distribution,        # {1: n, 2: n, 3: n}
            }
        }
        """
        if run_id is None:
            run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        timestamp = datetime.now(timezone.utc).isoformat()

        if country_list is None:
            country_list_path = SCHEMA_DIR / "target_countries.json"
            country_list = json.loads(country_list_path.read_text())

        results = []
        for entry in country_list:
            name = entry["country_name"]
            code = entry["country_code"]

            result = self.run_country(name, code, run_id, timestamp)
            results.append(result)

            status_icon = "✓" if result["status"] == "success" else "✗"
            print(
                f"  {status_icon} {code:4s} {name[:30]:30s} "
                f"attempts={result['attempts']}  status={result['status']}"
            )

            # Rate limit — avoid hammering the HiPerGator endpoint
            if self.request_delay > 0:
                time.sleep(self.request_delay)

        # --- Summary statistics ---
        succeeded = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        first_attempt_pass = sum(1 for r in succeeded if r["attempts"] == 1)
        total = len(results)

        retry_dist: dict[int, int] = {1: 0, 2: 0, 3: 0}
        for r in results:
            a = min(r["attempts"], 3)
            if a > 0:
                retry_dist[a] = retry_dist.get(a, 0) + 1

        avg_attempts = (
            sum(r["attempts"] for r in results) / total if total > 0 else 0.0
        )

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "constrained": self.constrained,
            "model": self.model,
            "temperature": self.temperature,
            "results": results,
            "summary": {
                "total": total,
                "succeeded": len(succeeded),
                "failed": len(failed),
                "schema_pass_rate": round(first_attempt_pass / total, 4) if total > 0 else 0.0,
                "avg_attempts": round(avg_attempts, 3),
                "retry_distribution": retry_dist,
            },
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the travel risk assessment pipeline.")
    parser.add_argument("--run-id", default=None, help="Override the auto-generated run ID")
    parser.add_argument("--unconstrained", action="store_true", help="Run without schema validation (baseline)")
    parser.add_argument("--country", default=None, help="Run for a single country code (e.g. FRA)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--output", default=None, help="Save run output JSON to this file path")
    args = parser.parse_args()

    orchestrator = Orchestrator(
        constrained=not args.unconstrained,
        request_delay=args.delay,
    )

    mode = "UNCONSTRAINED (baseline)" if args.unconstrained else "CONSTRAINED (schema validation + retry)"
    print(f"\n=== Pipeline run ===")
    print(f"Mode       : {mode}")
    print(f"Model      : {orchestrator.model}")
    print(f"Temperature: {orchestrator.temperature}")
    print(f"Max retries: {orchestrator.max_retries}")
    print()

    if args.country:
        # Single-country mode
        country_list_path = SCHEMA_DIR / "target_countries.json"
        all_countries = json.loads(country_list_path.read_text())
        match = next(
            (c for c in all_countries if c["country_code"].upper() == args.country.upper()),
            None,
        )
        if match is None:
            print(f"Country code '{args.country}' not found in target_countries.json")
            return
        run_id = args.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        result = orchestrator.run_country(match["country_name"], match["country_code"], run_id, timestamp)
        output = {"run_id": run_id, "timestamp": timestamp, "results": [result]}
    else:
        output = orchestrator.run_pipeline(run_id=args.run_id)

    # Print summary
    summary = output.get("summary", {})
    if summary:
        print(f"\n=== Run summary ({output['run_id']}) ===")
        print(f"  Total countries : {summary['total']}")
        print(f"  Succeeded       : {summary['succeeded']}")
        print(f"  Failed          : {summary['failed']}")
        print(f"  First-attempt pass rate : {summary['schema_pass_rate']:.1%}")
        print(f"  Avg LLM calls / country : {summary['avg_attempts']:.2f}")
        print(f"  Retry distribution      : {summary['retry_distribution']}")

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nOutput saved to {out_path}")
    else:
        print("\n" + json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
