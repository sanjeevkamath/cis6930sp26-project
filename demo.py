#!/usr/bin/env python3
"""
demo.py — Pre-recorded pipeline walkthrough for the Travel Risk Assessment project.

Covers four scenes in ~2 minutes:
  Scene 1  Frozen inputs      — show cached State Dept + news data
  Scene 2  Unconstrained x2  — run the LLM twice, highlight divergence
  Scene 3  Constrained        — schema validation + retry loop
  Scene 4  Stability results  — Jaccard score + full experiment results

Usage:
    uv run python demo.py
    uv run python demo.py --country COL
    uv run python demo.py --model llama-3.1-8b   # higher retry rate (~18%)
"""

import argparse
import textwrap
import uuid
from datetime import datetime, timezone

from src.pipeline.orchestrator import (
    Orchestrator,
    _load_cached_advisory,
    _load_cached_news,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 62
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(label: str) -> None:
    print(f"\n── {label} ──")


def _row_tokens(record: dict) -> frozenset:
    """Typed token set used by the stability metric."""
    tokens = set()
    for et in record.get("event_types", []):
        tokens.add(f"event_type:{et}")
    for rw in record.get("regional_warnings", []):
        tokens.add(f"regional_warning:{rw}")
    return frozenset(tokens)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _show_diff(tok_a: frozenset, tok_b: frozenset) -> None:
    shared  = sorted(tok_a & tok_b)
    only_a  = sorted(tok_a - tok_b)
    only_b  = sorted(tok_b - tok_a)
    print(f"  Shared     ({len(shared):2d}) : {shared or '—'}")
    print(f"  Only Run 1 ({len(only_a):2d}) : {only_a or '—'}")
    print(f"  Only Run 2 ({len(only_b):2d}) : {only_b or '—'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Demo walkthrough of the Travel Risk pipeline.")
    parser.add_argument("--country", default="MEX", metavar="CODE",
                        help="ISO 3166-1 alpha-3 country code (default: MEX)")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="Override model name (e.g. llama-3.1-8b for higher retry rate)")
    args = parser.parse_args()

    country_code = args.country.upper()
    model_kwargs = {"model": args.model} if args.model else {}

    # ── Scene 1: Frozen Inputs ────────────────────────────────────────────────
    _banner("SCENE 1 — FROZEN INPUTS")

    advisory = _load_cached_advisory(country_code)
    news     = _load_cached_news(country_code)

    if advisory is None:
        print(f"  ERROR: No cached advisory found for '{country_code}'.")
        print("  Run the state_dept_extract server first, or pick a different country.")
        return

    _section("State Department Advisory")
    print(f"  Country    : {advisory['country_name']} ({advisory['country_code']})")
    print(f"  Risk Level : {advisory['risk_level']} — {advisory['risk_label']}")
    summary_text = advisory.get("advisory_summary", "(none)")
    wrapped = textwrap.fill(
        summary_text, width=68,
        initial_indent="  Summary    : ",
        subsequent_indent="               ",
    )
    print(wrapped)

    _section("Latest news headline")
    if news:
        article = news.get("article", {})
        print(f"  Title  : {article.get('title', 'N/A')}")
        print(f"  Source : {article.get('source', 'N/A')}")
    else:
        print("  (no news cached for this country)")

    print()
    print("  Both sources are FROZEN at session start.")
    print("  Same data every run — only the model's sampling changes.")

    # ── Scene 2: Unconstrained runs ───────────────────────────────────────────
    _banner("SCENE 2 — UNCONSTRAINED BASELINE  (no schema guard, T=1.0)")

    base_orch = Orchestrator(constrained=False, request_delay=0.5, **model_kwargs)
    print(f"  Model       : {base_orch.model}")
    print(f"  Temperature : {base_orch.temperature}")
    print(f"  Constrained : False  →  raw output accepted as-is")

    run_id = f"demo-{uuid.uuid4().hex[:6]}"
    ts = datetime.now(timezone.utc).isoformat()

    print("\n  Running Run 1 …")
    r1 = base_orch.run_country(advisory["country_name"], country_code, f"{run_id}-u1", ts)
    print("  Running Run 2 …")
    r2 = base_orch.run_country(advisory["country_name"], country_code, f"{run_id}-u2", ts)

    rec1 = r1.get("record") or {}
    rec2 = r2.get("record") or {}

    _section("Run 1 structured output")
    print(f"  event_types       : {rec1.get('event_types', [])}")
    print(f"  regional_warnings : {rec1.get('regional_warnings', [])}")

    _section("Run 2 structured output")
    print(f"  event_types       : {rec2.get('event_types', [])}")
    print(f"  regional_warnings : {rec2.get('regional_warnings', [])}")

    tok1 = _row_tokens(rec1)
    tok2 = _row_tokens(rec2)
    jac  = _jaccard(tok1, tok2)

    _section("Token-level divergence")
    _show_diff(tok1, tok2)
    print(f"\n  Row Jaccard (Run 1 vs Run 2): {jac:.4f}")
    if jac < 1.0:
        print("  → Same input, different output. This is output volatility.")
    else:
        print("  → Runs matched exactly this time. Re-run to see divergence.")

    # ── Scene 3: Constrained run ──────────────────────────────────────────────
    _banner("SCENE 3 — CONSTRAINED GENERATION  (schema validation + retry)")

    con_orch = Orchestrator(constrained=True, request_delay=0.5, **model_kwargs)
    print(f"  Model       : {con_orch.model}")
    print(f"  Temperature : {con_orch.temperature}")
    print(f"  Constrained : True  →  validator fires on schema violations")
    print(f"  Max retries : {con_orch.max_retries}")

    print("\n  Running (constrained) …")
    r3 = con_orch.run_country(advisory["country_name"], country_code, f"{run_id}-c1", ts)
    rec3 = r3.get("record") or {}
    attempts = r3.get("attempts", 1)

    if attempts > 1:
        print(f"\n  ⚡ Validator fired — needed {attempts} attempt(s) to pass schema.")
        errors = r3.get("errors", [])
        if errors:
            for e in errors[:2]:
                print(f"     field={e.get('field')}  →  {e.get('message')}")
        print("     Error appended to context; model self-corrected on retry.")
    else:
        print("\n  ✓ Output passed schema validation on the first attempt.")

    _section("Constrained output")
    print(f"  event_types       : {rec3.get('event_types', [])}")
    print(f"  regional_warnings : {rec3.get('regional_warnings', [])}")
    print(f"  Status            : {r3.get('status', 'unknown')}")

    # ── Scene 4: Stability results ────────────────────────────────────────────
    _banner("SCENE 4 — STABILITY RESULTS")

    print(f"  This run's Jaccard (unconstrained Run1 vs Run2): {jac:.4f}")
    if tok1 | tok2:
        print(f"    |A ∩ B| = {len(tok1 & tok2)}   |A ∪ B| = {len(tok1 | tok2)}"
              f"   →  {len(tok1 & tok2)}/{len(tok1 | tok2)} = {jac:.4f}")

    print()
    print("  Full experiment: 30 countries × 5 runs × 2 models × 2 conditions")
    print()
    print("  ┌────────────────────────┬──────────────┬──────────────────┐")
    print("  │ Model / Condition      │ Mean Jaccard │ 95% Bootstrap CI │")
    print("  ├────────────────────────┼──────────────┼──────────────────┤")
    print("  │ llama-3.1-8b  baseline │    0.6016    │  [0.579, 0.623]  │")
    print("  │ llama-3.1-8b  +schema  │    0.6488    │  [0.623, 0.673]  │  +11.9%")
    print("  │ gpt-oss-120b  baseline │    0.4763    │  [0.303, 0.624]  │")
    print("  │ gpt-oss-120b  +schema  │    0.5016    │  [0.329, 0.641]  │  +4.8% (n.s.)")
    print("  └────────────────────────┴──────────────┴──────────────────┘")
    print()
    print("  Hypothesis: schema constraints reduce volatility by ≥ 50%.")
    print("  Best case : llama-3.1-8b → 11.9% improvement.")
    print("  Verdict   : HYPOTHESIS REJECTED.")
    print()
    print("  Surprise  : the 8B SLM is 26–29% MORE stable than the 120B LLM.")
    print("  Why       : canonicalization gap — 'Arauca' ≠ 'Arauca Department, Colombia'")
    print("              Schema validators are blind to semantic surface variation.")
    print()
    print("  Schema guardrails are necessary but not sufficient.")
    print("  Production IE pipelines need a canonicalization layer, not just retry loops.")


if __name__ == "__main__":
    main()