"""Fetch + TTL-cache READMEs for every starred repo, then clean them down to
a short plain-text intro suitable for embedding.

Cache semantics (per repo, keyed by `{owner}__{name}.json` under
data/cache/readmes/):
  - fresh entry (age < ttl_days)         -> skip entirely, zero API cost
  - stale entry with an etag             -> conditional GET; a 304 costs no
                                             rate-limit quota, just bumps
                                             fetched_at
  - missing entry                        -> full GET
  - 404 (no README)                      -> negative-cached so it isn't
                                             retried every run until it goes
                                             stale again
"""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .gh import GitHubClient, now_iso

CACHE_DIR = Path("data/cache/readmes")
MAX_CHARS = 1500


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

_HTML_BLOCK_RE = re.compile(r"<[^>]+>")
_BADGE_RE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")  # [![alt](img)](link)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INDENTED_CODE_RE = re.compile(r"(?:^|\n)(?: {4}|\t)[^\n]*")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


def clean_readme(raw: str) -> str:
    """Strip badges, HTML, code blocks, and boilerplate; keep the readable
    intro prose only, truncated to MAX_CHARS."""
    text = raw
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _BADGE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _FENCE_RE.sub(" ", text)
    text = _HTML_BLOCK_RE.sub(" ", text)
    text = _INDENTED_CODE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _TABLE_SEP_RE.sub(" ", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip(" |-*#>")
        if not stripped:
            continue
        if _PUNCT_ONLY_RE.match(stripped):
            continue
        lines.append(stripped)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_CHARS]


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _cache_path(owner: str, name: str) -> Path:
    return CACHE_DIR / f"{owner}__{name}.json"


def get_cached_readme(owner: str, name: str) -> str:
    """Public accessor for other pipeline stages (e.g. export.py) that just
    want whatever cleaned README text is on disk, without triggering a
    fetch."""
    entry = _load_cache(_cache_path(owner, name))
    return entry["text"] if entry else ""


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _is_fresh(entry: dict | None, ttl_days: int) -> bool:
    if entry is None:
        return False
    fetched_at = entry.get("fetched_at")
    if not fetched_at:
        return False
    age = datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)
    return age.days < ttl_days


def _fetch_one(gh: GitHubClient, owner: str, name: str, ttl_days: int) -> tuple[str, bool]:
    """Returns (cache_key, made_network_call)."""
    path = _cache_path(owner, name)
    entry = _load_cache(path)

    if _is_fresh(entry, ttl_days):
        return (path.name, False)

    headers = {}
    if entry and entry.get("etag"):
        headers["If-None-Match"] = entry["etag"]

    resp = gh.get(f"/repos/{owner}/{name}/readme", headers=headers)

    if resp.status_code == 304 and entry:
        entry["fetched_at"] = now_iso()
        path.write_text(json.dumps(entry))
        return (path.name, True)

    if resp.status_code == 404:
        path.write_text(
            json.dumps({"fetched_at": now_iso(), "etag": None, "text": "", "status": 404})
        )
        return (path.name, True)

    resp.raise_for_status()
    data = resp.json()
    raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    cleaned = clean_readme(raw)
    path.write_text(
        json.dumps(
            {
                "fetched_at": now_iso(),
                "etag": resp.headers.get("etag"),
                "text": cleaned,
                "status": 200,
            }
        )
    )
    return (path.name, True)


def enrich_readmes(
    stars_path: Path, ttl_days: int = 30, max_workers: int = 8
) -> dict[str, str]:
    """Ensure every repo in stars_path has a fresh-enough README cache entry.
    Returns a mapping of `owner/name` -> cleaned README text (possibly "")."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    repos = json.loads(stars_path.read_text())

    network_calls = 0
    with GitHubClient() as gh, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, gh, r["owner"], r["name"], ttl_days): r
            for r in repos
        }
        failures = 0
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="enriching READMEs"
        ):
            repo = futures[fut]
            try:
                _, made_call = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad repo shouldn't kill the run
                failures += 1
                tqdm.write(f"[enrich] failed {repo['owner']}/{repo['name']}: {e}")
                continue
            if made_call:
                network_calls += 1

    if failures:
        print(f"[enrich] {failures} repos failed and were left with stale/no cache")

    print(f"[enrich] {network_calls}/{len(repos)} repos required a network call")

    result: dict[str, str] = {}
    for r in repos:
        entry = _load_cache(_cache_path(r["owner"], r["name"]))
        result[f"{r['owner']}/{r['name']}"] = entry["text"] if entry else ""
    return result
