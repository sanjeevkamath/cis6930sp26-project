"""
NewsExtractServer — MCP server for fetching news articles via NewsAPI.

Free tier limit: 100 requests/day. This server fetches exactly 1 article per
country (30 total) and caches results immediately to data/raw/news/ so the
API is never called again for cached countries.

Articles introduce realistic input noise for the stability experiments:
repeated pipeline runs use the same cached articles, while perturbation
experiments swap one article in/out per country.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
CACHE_DIR = Path(__file__).parents[2] / "data" / "raw" / "news"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TravelAdvisoryResearchBot/1.0)"}

mcp = FastMCP("NewsExtractServer")


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


def _fetch_article(country_name: str) -> dict | None:
    """
    Fetch 1 recent English-language article about the country from NewsAPI.
    Returns the article dict or None if no results.
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not set in environment")

    params = {
        "q": f'"{country_name}"',
        "language": "en",
        "pageSize": 1,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }

    resp = requests.get(NEWSAPI_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", [])
    return articles[0] if articles else None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_news(country_name: str, country_code: str) -> dict | None:
    """
    Fetch and cache 1 news article for a country.

    Checks local cache first — no API call is made if already cached.

    Args:
        country_name: Country name used as the search query (e.g. "France").
        country_code: ISO 3166-1 alpha-3 code used as the cache key (e.g. "FRA").

    Returns a dict with:
        country_code, country_name, article (title, description, url,
        source, published_at), fetched_at
    Or None if no article was found.
    """
    cached = _load_from_cache(country_code)
    if cached:
        return cached

    raw = _fetch_article(country_name)
    if raw is None:
        return None

    result = {
        "country_code": country_code.upper(),
        "country_name": country_name,
        "article": {
            "title": raw.get("title", ""),
            "description": raw.get("description", ""),
            "content": (raw.get("content") or "")[:1000],
            "url": raw.get("url", ""),
            "source": (raw.get("source") or {}).get("name", ""),
            "published_at": raw.get("publishedAt", ""),
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    _save_to_cache(country_code, result)
    return result


@mcp.tool()
def fetch_news_for_all(country_list: list[dict], delay_seconds: float = 1.5) -> dict:
    """
    Fetch and cache news for a list of countries, skipping already-cached ones.

    Budget-aware: only makes API calls for countries without a cached file.
    With 30 countries this costs at most 30 of the 100 daily requests.

    Args:
        country_list: List of { country_name, country_code } dicts.
        delay_seconds: Pause between API calls to avoid rate limiting.

    Returns:
        { fetched: [...], skipped_cached: [...], failed: [...], api_calls_made: int }
    """
    fetched = []
    skipped = []
    failed = []
    api_calls = 0

    for entry in country_list:
        name = entry.get("country_name", "")
        code = entry.get("country_code", "")

        if not name or not code:
            failed.append({"entry": entry, "error": "Missing country_name or country_code"})
            continue

        # Skip if already cached — no API call
        if _load_from_cache(code) is not None:
            skipped.append(code)
            continue

        try:
            result = fetch_news(name, code)
            if result:
                fetched.append(result)
            else:
                failed.append({"entry": entry, "error": "No articles found"})
            api_calls += 1
            time.sleep(delay_seconds)
        except Exception as exc:
            failed.append({"entry": entry, "error": str(exc)})

    return {
        "fetched": fetched,
        "skipped_cached": skipped,
        "failed": failed,
        "api_calls_made": api_calls,
    }


@mcp.tool()
def get_cached_news(country_code: str) -> dict | None:
    """Return cached news for the given ISO alpha-3 code, or None."""
    return _load_from_cache(country_code)


@mcp.tool()
def list_cached_countries() -> list[str]:
    """Return the list of country codes that have cached news articles."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in sorted(CACHE_DIR.glob("*.json"))]


if __name__ == "__main__":
    mcp.run()
