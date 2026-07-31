"""Final pipeline stage: assemble everything (stars, embeddings, topics,
related repos, README excerpts) into the JSON + binary assets the Astro
site reads from site/src/data/ and site/public/search/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .readmes import get_cached_readme
from .related import compute_related

SITE_DATA_DIR = Path("site/src/data")
SITE_SEARCH_DIR = Path("site/public/search")

MIN_TAG_SIZE = 3
README_EXCERPT_CHARS = 400


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)


def _build_tags(repos: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Returns (tags_by_slug, repo_id -> [tag_slug]).

    GitHub topics and the primary language share one namespace: a repo
    tagged `rust` and one written in language `Rust` land in the same
    `/tag/rust` bucket, since to a browsing human they mean the same thing.
    """
    buckets: dict[str, dict] = {}  # slug -> {label, repo_ids: set}

    def add(repo_id: str, raw_label: str) -> None:
        slug = _slugify(raw_label)
        if not slug:
            return
        bucket = buckets.setdefault(slug, {"label": raw_label, "repo_ids": set()})
        # GitHub topic strings are always lowercase; language names carry
        # proper casing ("TypeScript", "Rust"). Prefer whichever label isn't
        # all-lowercase so tags don't render as "rust" in one place and
        # "TypeScript" in another depending on which repo hit first.
        if bucket["label"].islower() and not raw_label.islower():
            bucket["label"] = raw_label
        bucket["repo_ids"].add(repo_id)

    for r in repos:
        rid = str(r["id"])
        for t in r.get("topics") or []:
            add(rid, t)
        if r.get("language"):
            add(rid, r["language"])

    tags_by_slug = {
        slug: {"label": b["label"], "repo_ids": sorted(b["repo_ids"])}
        for slug, b in buckets.items()
        if len(b["repo_ids"]) >= MIN_TAG_SIZE
    }

    repo_tags: dict[str, list[str]] = {str(r["id"]): [] for r in repos}
    for slug, b in tags_by_slug.items():
        for rid in b["repo_ids"]:
            repo_tags[rid].append(slug)

    return tags_by_slug, repo_tags


def _parent_group_labels(topics: list[dict]) -> dict[int, str]:
    from collections import Counter

    groups: dict[int, Counter] = {}
    sizes: dict[int, int] = {}
    for t in topics:
        pg = t["parent_group"]
        groups.setdefault(pg, Counter())
        for kw in t["keywords"][:5]:
            if kw.isdigit() or len(kw) <= 1:
                continue
            groups[pg][kw] += t["size"]
        sizes[pg] = sizes.get(pg, 0) + t["size"]

    labels = {}
    for pg, counter in groups.items():
        top = [w for w, _ in counter.most_common(2)]
        labels[pg] = " & ".join(w.title() for w in top) if top else f"Group {pg}"
    return labels


def _quantize_int8(matrix: np.ndarray) -> np.ndarray:
    """Symmetric int8 quantization. Rows are already L2-normalized (values
    in [-1, 1]), so a single fixed scale of 127 works for every row without
    per-vector scale factors."""
    clipped = np.clip(matrix, -1.0, 1.0)
    return np.round(clipped * 127).astype(np.int8)


def export_site_data(
    stars_path: Path,
    embeddings_path: Path,
    embeddings_meta_path: Path,
    topics_result_path: Path,
) -> None:
    repos = json.loads(stars_path.read_text())
    matrix = np.load(embeddings_path)
    emb_meta = json.loads(embeddings_meta_path.read_text())
    order: list[str] = emb_meta["order"]
    topics_result = json.loads(topics_result_path.read_text())

    id_to_repo = {str(r["id"]): r for r in repos}
    assignments: dict[str, str | None] = topics_result["assignments"]
    topics: list[dict] = topics_result["topics"]
    parent_labels = _parent_group_labels(topics)

    tags_by_slug, repo_tags = _build_tags(repos)
    related = compute_related(embeddings_path, embeddings_meta_path)

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SITE_SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # --- repos.json ---------------------------------------------------
    repos_out = []
    for rid in order:
        r = id_to_repo[rid]
        readme = get_cached_readme(r["owner"], r["name"])
        repos_out.append(
            {
                **r,
                "id": rid,
                "topic_slug": assignments.get(rid),
                "tags": repo_tags.get(rid, []),
                "related": related.get(rid, []),
                "readme_excerpt": readme[:README_EXCERPT_CHARS],
            }
        )
    (SITE_DATA_DIR / "repos.json").write_text(
        json.dumps(repos_out, ensure_ascii=False)
    )

    # --- topics.json ----------------------------------------------------
    topics_out = [
        {
            "slug": t["slug"],
            "label": t["label"],
            "keywords": t["keywords"],
            "size": t["size"],
            "parent_group": t["parent_group"],
            "parent_label": parent_labels.get(t["parent_group"], ""),
            "repo_ids": t["repo_ids"],
        }
        for t in topics
    ]
    topics_out.sort(key=lambda t: -t["size"])
    (SITE_DATA_DIR / "topics.json").write_text(
        json.dumps(topics_out, ensure_ascii=False)
    )

    # --- tags.json --------------------------------------------------------
    tags_out = [
        {"slug": slug, "label": b["label"], "repo_ids": b["repo_ids"]}
        for slug, b in sorted(
            tags_by_slug.items(), key=lambda kv: -len(kv[1]["repo_ids"])
        )
    ]
    (SITE_DATA_DIR / "tags.json").write_text(json.dumps(tags_out, ensure_ascii=False))

    # --- stats.json ---------------------------------------------------
    from collections import Counter

    lang_counts = Counter(r.get("language") or "Unknown" for r in repos)
    license_counts = Counter(r.get("license") or "None" for r in repos)
    by_month = Counter(
        (r.get("starred_at") or "")[:7] for r in repos if r.get("starred_at")
    )
    stats = {
        "total_repos": len(repos),
        "total_topics": len(topics_out),
        "total_tags": len(tags_out),
        "archived_count": sum(1 for r in repos if r.get("archived")),
        "fork_count": sum(1 for r in repos if r.get("fork")),
        "language_histogram": dict(lang_counts.most_common(20)),
        "license_histogram": dict(license_counts.most_common(10)),
        "stars_by_month": dict(sorted(by_month.items())),
    }
    (SITE_DATA_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False))

    # --- search/index.json (lexical) ---------------------------------------
    search_index = [
        {
            "id": rid,
            "full_name": id_to_repo[rid]["full_name"],
            "description": id_to_repo[rid].get("description") or "",
            "tags": repo_tags.get(rid, []),
            "topic_slug": assignments.get(rid),
        }
        for rid in order
    ]
    (SITE_SEARCH_DIR / "index.json").write_text(
        json.dumps(search_index, ensure_ascii=False)
    )

    # --- search/vectors.bin + vectors.json (semantic) -----------------------
    quantized = _quantize_int8(matrix)
    (SITE_SEARCH_DIR / "vectors.bin").write_bytes(quantized.tobytes())
    (SITE_SEARCH_DIR / "vectors.json").write_text(
        json.dumps({"n": len(order), "dim": matrix.shape[1], "order": order})
    )

    print(
        f"[export] {len(repos_out)} repos, {len(topics_out)} topics, "
        f"{len(tags_out)} tags -> {SITE_DATA_DIR}, {SITE_SEARCH_DIR}"
    )
