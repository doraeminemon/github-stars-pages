"""Fetch the full starred-repo list for a GitHub user.

Uses the `application/vnd.github.star+json` Accept header, which is the
only way to get `starred_at` back from the API — the plain
`application/vnd.github+json` accept silently omits it.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .gh import GitHubClient

STAR_ACCEPT = "application/vnd.github.star+json"

# Fields we actually use on the site; discards the ~40 *_url fields GitHub
# returns per repo.
_KEEP_REPO_FIELDS = (
    "id",
    "full_name",
    "name",
    "description",
    "html_url",
    "homepage",
    "language",
    "stargazers_count",
    "forks_count",
    "open_issues_count",
    "topics",
    "archived",
    "fork",
    "created_at",
    "pushed_at",
)


def _slim(entry: dict) -> dict:
    repo = entry["repo"]
    out = {k: repo.get(k) for k in _KEEP_REPO_FIELDS}
    out["owner"] = repo["owner"]["login"]
    license_ = repo.get("license") or {}
    out["license"] = license_.get("spdx_id")
    out["starred_at"] = entry.get("starred_at")
    return out


def fetch_stars(login: str, out_path: Path) -> list[dict]:
    with GitHubClient(accept=STAR_ACCEPT) as gh:
        items = []
        for entry in tqdm(
            gh.paginate(f"/users/{login}/starred"), desc="fetching stars", unit="repo"
        ):
            items.append(_slim(entry))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"[fetch] wrote {len(items)} starred repos -> {out_path}")
    return items
