"""Shared GitHub API client.

Authenticates once via `gh auth token` (or the GH_TOKEN / GITHUB_TOKEN env
vars, used in CI) and drives a single reusable httpx.Client rather than
shelling out to `gh api` per-request. This gives us connection pooling,
easy concurrency, and centralized rate-limit handling.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone

import httpx

API_BASE = "https://api.github.com"


def _resolve_token() -> str:
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "No GitHub token found. Set GH_TOKEN/GITHUB_TOKEN or run `gh auth login`."
        ) from e


class GitHubClient:
    """A rate-limit-aware wrapper around httpx.Client for the GitHub REST API."""

    def __init__(self, accept: str = "application/vnd.github+json") -> None:
        token = _resolve_token()
        self._client = httpx.Client(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self, resp: httpx.Response) -> None:
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if remaining is not None and int(remaining) < 50 and reset:
            wait = int(reset) - int(time.time()) + 2
            if wait > 0:
                print(f"[gh] rate limit low ({remaining} left), sleeping {wait}s")
                time.sleep(wait)

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Issue a request with retry on 5xx / secondary rate limits."""
        backoff = 2.0
        for attempt in range(6):
            resp = self._client.request(method, path, **kwargs)
            if resp.status_code == 403 and (
                "rate limit" in resp.text.lower() or "abuse" in resp.text.lower()
            ):
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else backoff
                print(f"[gh] secondary rate limit hit, sleeping {wait}s")
                time.sleep(wait)
                backoff *= 2
                continue
            if resp.status_code >= 500:
                print(f"[gh] {resp.status_code} on {path}, retrying in {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            self._throttle(resp)
            return resp
        resp.raise_for_status()
        return resp

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def paginate(self, path: str, params: dict | None = None):
        """Yield JSON items across all pages of a paginated endpoint."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        next_url: str | None = path
        next_params: dict | None = params
        while next_url:
            resp = self.get(next_url, params=next_params)
            resp.raise_for_status()
            yield from resp.json()
            next_url = None
            next_params = None
            link = resp.headers.get("link")
            if link:
                for part in link.split(","):
                    if 'rel="next"' in part:
                        url = part[part.find("<") + 1 : part.find(">")]
                        # Subsequent requests carry their own query string.
                        next_url = url.replace(API_BASE, "")
                        next_params = None
                        break


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
