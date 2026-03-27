"""
LoadServer — MCP server for persisting validated travel risk records to SQLite.

Responsibilities:
  - Insert validated records from the LLM orchestrator into a SQLite database.
  - Create versioned run snapshots tagged with run_id and timestamp.
  - Expose snapshot listing and retrieval for the StabilityAnalysisServer.

Database layout
---------------
  records
    id           INTEGER PRIMARY KEY
    run_id       TEXT NOT NULL
    country_code TEXT NOT NULL
    risk_level   INTEGER
    risk_label   TEXT
    event_types  TEXT   (JSON array serialized as string)
    record_json  TEXT   (full validated record as JSON string)
    inserted_at  TEXT   (ISO 8601 UTC)
    UNIQUE(run_id, country_code)  -- idempotent upsert; safe to re-run

  run_snapshots
    run_id         TEXT PRIMARY KEY
    timestamp      TEXT
    constrained    INTEGER  (1 = constrained, 0 = unconstrained baseline)
    model          TEXT
    total          INTEGER
    succeeded      INTEGER
    failed         INTEGER
    schema_pass_rate  REAL
    avg_attempts   REAL
    summary_json   TEXT  (full summary dict as JSON string)
    created_at     TEXT
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parents[2] / "data" / "travel_advisory.db"

mcp = FastMCP("LoadServer")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT NOT NULL,
            country_code TEXT NOT NULL,
            risk_level   INTEGER,
            risk_label   TEXT,
            event_types  TEXT,
            record_json  TEXT NOT NULL,
            inserted_at  TEXT NOT NULL,
            UNIQUE(run_id, country_code)
        );

        CREATE TABLE IF NOT EXISTS run_snapshots (
            run_id            TEXT PRIMARY KEY,
            timestamp         TEXT,
            constrained       INTEGER,
            model             TEXT,
            total             INTEGER,
            succeeded         INTEGER,
            failed            INTEGER,
            schema_pass_rate  REAL,
            avg_attempts      REAL,
            summary_json      TEXT,
            created_at        TEXT NOT NULL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def insert_record(record: dict, run_id: str) -> dict:
    """
    Insert a single validated travel risk record into the database.

    Performs an upsert on (run_id, country_code) — safe to call multiple times
    for the same country/run without creating duplicates.

    Args:
        record: Validated JSON record from the LLM orchestrator.
        run_id: Identifier for the pipeline run that produced this record.

    Returns:
        { inserted: bool, country_code: str, run_id: str }
    """
    country_code = record.get("country_code", "")
    if not country_code:
        return {"inserted": False, "error": "record missing country_code"}

    now = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO records
                (run_id, country_code, risk_level, risk_label, event_types, record_json, inserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, country_code) DO UPDATE SET
                risk_level  = excluded.risk_level,
                risk_label  = excluded.risk_label,
                event_types = excluded.event_types,
                record_json = excluded.record_json,
                inserted_at = excluded.inserted_at
            """,
            (
                run_id,
                country_code,
                record.get("risk_level"),
                record.get("risk_label"),
                json.dumps(record.get("event_types", [])),
                json.dumps(record),
                now,
            ),
        )
        conn.commit()

    return {"inserted": True, "country_code": country_code, "run_id": run_id}


@mcp.tool()
def insert_run_results(pipeline_output: dict) -> dict:
    """
    Persist all successful records from a full pipeline run in one call.

    Args:
        pipeline_output: The dict returned by Orchestrator.run_pipeline().

    Returns:
        { run_id, inserted, skipped_failed, snapshot_created }
    """
    run_id = pipeline_output.get("run_id", "")
    results = pipeline_output.get("results", [])

    inserted = 0
    skipped = 0
    for result in results:
        if result.get("status") == "success" and result.get("record"):
            insert_record(result["record"], run_id)
            inserted += 1
        else:
            skipped += 1

    # Create the run snapshot
    snapshot = create_snapshot(pipeline_output)

    return {
        "run_id": run_id,
        "inserted": inserted,
        "skipped_failed": skipped,
        "snapshot_created": snapshot.get("created", False),
    }


@mcp.tool()
def create_snapshot(pipeline_output: dict) -> dict:
    """
    Record run-level metadata as a versioned snapshot.

    Args:
        pipeline_output: The dict returned by Orchestrator.run_pipeline().

    Returns:
        { created: bool, run_id: str }
    """
    run_id = pipeline_output.get("run_id", "")
    if not run_id:
        return {"created": False, "error": "pipeline_output missing run_id"}

    summary = pipeline_output.get("summary", {})
    now = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO run_snapshots
                (run_id, timestamp, constrained, model, total, succeeded, failed,
                 schema_pass_rate, avg_attempts, summary_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                pipeline_output.get("timestamp"),
                1 if pipeline_output.get("constrained", True) else 0,
                pipeline_output.get("model"),
                summary.get("total"),
                summary.get("succeeded"),
                summary.get("failed"),
                summary.get("schema_pass_rate"),
                summary.get("avg_attempts"),
                json.dumps(summary),
                now,
            ),
        )
        conn.commit()

    return {"created": True, "run_id": run_id}


@mcp.tool()
def list_snapshots() -> list[dict]:
    """
    List all stored run snapshots ordered by creation time.

    Returns a list of snapshot metadata dicts (excludes full record data).
    """
    with _get_connection() as conn:
        _ensure_tables(conn)
        rows = conn.execute(
            """
            SELECT run_id, timestamp, constrained, model, total, succeeded,
                   failed, schema_pass_rate, avg_attempts, created_at
            FROM run_snapshots
            ORDER BY created_at ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


@mcp.tool()
def get_snapshot_records(run_id: str) -> list[dict]:
    """
    Return all records stored for a given run_id.

    Args:
        run_id: The run identifier.

    Returns:
        List of full record dicts for that run.
    """
    with _get_connection() as conn:
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT record_json FROM records WHERE run_id = ? ORDER BY country_code",
            (run_id,),
        ).fetchall()

    return [json.loads(row["record_json"]) for row in rows]


@mcp.tool()
def get_record(run_id: str, country_code: str) -> dict | None:
    """
    Return the stored record for a specific run and country.

    Args:
        run_id: The run identifier.
        country_code: ISO 3166-1 alpha-3 country code.

    Returns:
        The full record dict, or None if not found.
    """
    with _get_connection() as conn:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT record_json FROM records WHERE run_id = ? AND country_code = ?",
            (run_id, country_code.upper()),
        ).fetchone()

    return json.loads(row["record_json"]) if row else None


@mcp.tool()
def list_run_ids() -> list[str]:
    """Return all stored run IDs ordered by creation time."""
    with _get_connection() as conn:
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT run_id FROM run_snapshots ORDER BY created_at ASC"
        ).fetchall()
    return [row["run_id"] for row in rows]


if __name__ == "__main__":
    mcp.run()
