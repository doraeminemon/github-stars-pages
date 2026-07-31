"""Compute top-k related repos for every starred repo via cosine similarity
over the (already L2-normalized) embedding matrix.

At ~2000 repos the full pairwise similarity matrix is tiny (~15MB as
float32), so this is a single dense matmul rather than an approximate
nearest-neighbor index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TOP_K = 12


def compute_related(
    embeddings_path: Path, embeddings_meta_path: Path, top_k: int = TOP_K
) -> dict[str, list[dict]]:
    matrix = np.load(embeddings_path)
    meta = json.loads(embeddings_meta_path.read_text())
    order: list[str] = meta["order"]

    # Rows are already normalized by embed.py (normalize_embeddings=True),
    # but re-normalize defensively in case that ever changes upstream.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms

    sims = normed @ normed.T
    n = len(order)
    related: dict[str, list[dict]] = {}

    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf  # exclude self
        k = min(top_k, n - 1)
        if k <= 0:
            related[order[i]] = []
            continue
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        related[order[i]] = [
            {"repo_id": order[j], "score": round(float(row[j]), 4)} for j in top_idx
        ]

    return related
