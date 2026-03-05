"""
SchemaValidationServer — MCP server for validating LLM-generated travel advisory records.

This is the core architectural constraint under study. Every row produced by the LLM
orchestrator must pass through this server before being accepted into the pipeline.
On failure, structured error feedback is returned so the orchestrator can re-prompt
the LLM (up to MAX_RETRIES times).
"""

import json
from pathlib import Path

import jsonschema
from jsonschema import Draft7Validator
from mcp.server.fastmcp import FastMCP

SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "travel_advisory.json"
MAX_RETRIES = 3

mcp = FastMCP("SchemaValidationServer")

# Load schema once at module import time
with SCHEMA_PATH.open() as _f:
    _SCHEMA = json.load(_f)

_VALIDATOR = Draft7Validator(_SCHEMA)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_errors(instance: dict) -> list[dict]:
    """
    Run the JSON Schema validator and return a list of structured error dicts.
    Each dict has: { field, message, invalid_value }.
    """
    errors = []
    for error in sorted(_VALIDATOR.iter_errors(instance), key=lambda e: str(e.path)):
        field = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(
            {
                "field": field,
                "message": error.message,
                "invalid_value": repr(error.instance),
            }
        )
    return errors


def _build_retry_prompt(errors: list[dict]) -> str:
    """
    Produce a concise prompt fragment that instructs the LLM exactly what to fix.
    This is appended to the original prompt when retrying.
    """
    lines = ["The previous output failed schema validation. Fix the following issues:"]
    for i, err in enumerate(errors, 1):
        lines.append(f"  {i}. Field '{err['field']}': {err['message']}")
    lines.append(
        "\nReturn ONLY valid JSON matching the schema. "
        "Do not include any explanation or markdown."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def validate_record(record: dict) -> dict:
    """
    Validate a single travel advisory record against the schema.

    Args:
        record: The LLM-generated JSON object to validate.

    Returns:
        {
            valid: bool,
            errors: list[{ field, message, invalid_value }],
            retry_prompt: str | None   # non-null only when valid=False
        }
    """
    errors = _collect_errors(record)
    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "retry_prompt": _build_retry_prompt(errors) if not valid else None,
    }


@mcp.tool()
def validate_batch(records: list[dict]) -> dict:
    """
    Validate a batch of records and return per-record results plus aggregate stats.

    Args:
        records: List of LLM-generated JSON objects.

    Returns:
        {
            results: list[{ index, valid, errors, retry_prompt }],
            total: int,
            passed: int,
            failed: int,
            pass_rate: float
        }
    """
    results = []
    passed = 0

    for i, record in enumerate(records):
        errors = _collect_errors(record)
        valid = len(errors) == 0
        if valid:
            passed += 1
        results.append(
            {
                "index": i,
                "valid": valid,
                "errors": errors,
                "retry_prompt": _build_retry_prompt(errors) if not valid else None,
            }
        )

    total = len(records)
    return {
        "results": results,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
    }


@mcp.tool()
def get_schema() -> dict:
    """Return the full JSON Schema used for validation."""
    return _SCHEMA


@mcp.tool()
def get_field_constraints() -> dict:
    """
    Return a concise summary of field constraints for use in LLM prompts.
    Helps the orchestrator include schema context without sending the full schema.
    """
    props = _SCHEMA.get("properties", {})
    required = _SCHEMA.get("required", [])

    summary = {}
    for field, spec in props.items():
        entry: dict = {"required": field in required}
        if "type" in spec:
            entry["type"] = spec["type"]
        if "enum" in spec:
            entry["allowed_values"] = spec["enum"]
        if "minimum" in spec:
            entry["minimum"] = spec["minimum"]
        if "maximum" in spec:
            entry["maximum"] = spec["maximum"]
        if "pattern" in spec:
            entry["pattern"] = spec["pattern"]
        if "minLength" in spec:
            entry["minLength"] = spec["minLength"]
        if "items" in spec and "enum" in spec["items"]:
            entry["allowed_items"] = spec["items"]["enum"]
        summary[field] = entry

    return summary


if __name__ == "__main__":
    mcp.run()
