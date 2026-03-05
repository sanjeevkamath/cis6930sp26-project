"""
StateDeptExtractServer — MCP server for fetching U.S. State Department travel advisories.

Uses the official RSS feed (no API key, no CAPTCHA):
  https://travel.state.gov/_res/rss/TAsTWs.xml

The feed returns all 213+ country advisories in a single request. Results are cached
locally as JSON files under data/raw/state_dept/ to ensure reproducibility regardless
of network availability during subsequent pipeline runs.
"""

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_URL = "https://travel.state.gov/_res/rss/TAsTWs.xml"
CACHE_DIR = Path(__file__).parents[2] / "data" / "raw" / "state_dept"

RISK_LABELS = {
    1: "Exercise Normal Precautions",
    2: "Exercise Increased Caution",
    3: "Reconsider Travel",
    4: "Do Not Travel",
}

# Keyword → event_type taxonomy mapping (order matters: more specific first)
EVENT_KEYWORDS: list[tuple[str, str]] = [
    (r"kidnap", "kidnapping"),
    (r"terrorism|terrorist", "terrorism"),
    (r"civil unrest|political unrest|protest|demonstration|riot", "civil_unrest"),
    (r"armed conflict|war|military|combat|hostil", "conflict"),
    (r"crime|robbery|theft|assault|murder|homicide", "crime"),
    (r"health|disease|epidemic|pandemic|medical|cholera|malaria|dengue", "health"),
    (r"earthquake|hurricane|flood|volcano|tsunami|natural disaster|cyclone", "natural_disaster"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TravelAdvisoryResearchBot/1.0)"
    )
}

mcp = FastMCP("StateDeptExtractServer")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(country_code: str) -> Path:
    return CACHE_DIR / f"{country_code.upper()}.json"


def _load_from_cache(country_code: str) -> dict | None:
    path = _cache_path(country_code)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def _save_to_cache(country_code: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _cache_path(country_code).open("w") as f:
        json.dump(data, f, indent=2)


def _parse_title(title: str) -> tuple[str, int, str]:
    """
    Parse an RSS title like 'France - Level 2: Exercise Increased Caution'.
    Returns (country_name, risk_level, risk_label).
    """
    m = re.match(r"^(.+?)\s*[-–]\s*Level\s+(\d)\s*:\s*(.+)$", title, re.IGNORECASE)
    if m:
        country_name = m.group(1).strip()
        risk_level = int(m.group(2))
        risk_label = m.group(3).strip()
        return country_name, risk_level, risk_label
    return title.strip(), 1, RISK_LABELS[1]


def _extract_event_types(text: str) -> list[str]:
    """Extract event_type labels by matching keywords against advisory text."""
    found = []
    lower = text.lower()
    for pattern, label in EVENT_KEYWORDS:
        if re.search(pattern, lower) and label not in found:
            found.append(label)
    return found if found else ["other"]


def _normalize(name: str) -> str:
    """Lowercase and strip non-ASCII characters for fuzzy name matching."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


def _extract_country_code_from_url(url: str) -> str | None:
    """
    Extract ISO alpha-3 code from newer State Dept URL format:
      .../destination.pak.html  →  PAK
    """
    m = re.search(r"destination\.([a-z]{3})\.html", url, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _fetch_rss() -> list[dict]:
    """
    Fetch and parse the RSS feed. Returns a list of raw advisory dicts.
    """
    resp = requests.get(RSS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml-xml")
    items = soup.find_all("item")

    advisories = []
    for item in items:
        title_tag = item.find("title")
        link_tag = item.find("link")
        desc_tag = item.find("description")
        date_tag = item.find("pubDate")

        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        link = link_tag.get_text(strip=True) if link_tag else ""
        pub_date = date_tag.get_text(strip=True) if date_tag else ""

        # Parse description HTML
        desc_html = desc_tag.get_text(strip=True) if desc_tag else ""
        desc_text = BeautifulSoup(desc_html, "lxml").get_text(separator=" ", strip=True)

        country_name, risk_level, risk_label = _parse_title(title)

        advisories.append(
            {
                "country_name": country_name,
                "risk_level": risk_level,
                "risk_label": risk_label,
                "advisory_url": link,
                "description": desc_text[:3000],
                "pub_date": pub_date,
            }
        )

    return advisories


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_advisories() -> list[dict]:
    """
    Fetch all country travel advisories from the State Dept RSS feed.

    Returns a list of objects:
        { country_name, risk_level, risk_label, advisory_url, description, pub_date }

    One HTTP request for all 200+ countries. Not cached at this level.
    """
    return _fetch_rss()


@mcp.tool()
def fetch_advisory(country_name: str, country_code: str) -> dict | None:
    """
    Fetch and cache the travel advisory for a single country by matching
    its name against the RSS feed.

    Args:
        country_name: Country name as it appears in the State Dept feed (e.g. "France").
        country_code: ISO 3166-1 alpha-3 code used as the cache key (e.g. "FRA").

    Returns a structured advisory dict, or None if the country is not found in the feed.
    """
    cached = _load_from_cache(country_code)
    if cached:
        return cached

    advisories = _fetch_rss()
    match = None
    target = _normalize(country_name)
    for entry in advisories:
        if _normalize(entry["country_name"]) == target:
            match = entry
            break

    if match is None:
        return None

    # Resolve country code: prefer code extracted from URL, fall back to argument
    url_code = _extract_country_code_from_url(match["advisory_url"])
    resolved_code = url_code or country_code.upper()

    event_types = _extract_event_types(match["description"])

    result = {
        "country_name": match["country_name"],
        "country_code": resolved_code,
        "risk_level": match["risk_level"],
        "risk_label": match["risk_label"],
        "advisory_summary": match["description"],
        "regional_warnings": [],   # populated by LLM orchestrator from full text
        "entry_exit_requirements": "",  # populated by LLM orchestrator
        "event_types": event_types,
        "advisory_url": match["advisory_url"],
        "pub_date": match["pub_date"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    _save_to_cache(resolved_code, result)
    return result


@mcp.tool()
def fetch_target_countries(country_list: list[dict]) -> dict:
    """
    Fetch and cache advisories for a defined list of target countries.

    Args:
        country_list: List of { country_name, country_code } dicts.

    Returns:
        { fetched: [...], failed: [...], risk_distribution: {1:n, 2:n, 3:n, 4:n} }
    """
    advisories = _fetch_rss()
    index = {_normalize(entry["country_name"]): entry for entry in advisories}

    fetched = []
    failed = []

    for target in country_list:
        name = target.get("country_name", "")
        code = target.get("country_code", "")

        if not name or not code:
            failed.append({"entry": target, "error": "Missing country_name or country_code"})
            continue

        # Check cache first
        cached = _load_from_cache(code)
        if cached:
            fetched.append(cached)
            continue

        match = index.get(_normalize(name))
        if match is None:
            failed.append({"entry": target, "error": f"'{name}' not found in RSS feed"})
            continue

        url_code = _extract_country_code_from_url(match["advisory_url"])
        resolved_code = url_code or code.upper()
        event_types = _extract_event_types(match["description"])

        result = {
            "country_name": match["country_name"],
            "country_code": resolved_code,
            "risk_level": match["risk_level"],
            "risk_label": match["risk_label"],
            "advisory_summary": match["description"],
            "regional_warnings": [],
            "entry_exit_requirements": "",
            "event_types": event_types,
            "advisory_url": match["advisory_url"],
            "pub_date": match["pub_date"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        _save_to_cache(resolved_code, result)
        fetched.append(result)

    risk_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in fetched:
        level = r.get("risk_level", 0)
        if level in risk_dist:
            risk_dist[level] += 1

    return {
        "fetched": fetched,
        "failed": failed,
        "risk_distribution": risk_dist,
    }


@mcp.tool()
def get_cached_advisory(country_code: str) -> dict | None:
    """Return a cached advisory for the given ISO alpha-3 code, or None."""
    return _load_from_cache(country_code)


@mcp.tool()
def list_cached_countries() -> list[str]:
    """Return the list of cached ISO alpha-3 country codes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in sorted(CACHE_DIR.glob("*.json"))]


if __name__ == "__main__":
    mcp.run()
