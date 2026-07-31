"""Build sentence embeddings for every starred repo.

Uses `sentence-transformers/all-MiniLM-L6-v2` specifically because
`Xenova/all-MiniLM-L6-v2` (used client-side by the search island) is the
exact ONNX export of the same model — Python-side and browser-side vectors
live in the same 384-dim space, which the hybrid search in the Astro site
depends on.

Embeddings are cached in data/embeddings.npy + data/embeddings_meta.json,
keyed by a hash of each repo's embedding document, so re-running only
re-encodes repos whose description/topics/README actually changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .readmes import enrich_readmes

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDINGS_PATH = Path("data/embeddings.npy")
META_PATH = Path("data/embeddings_meta.json")


def build_doc(repo: dict, readme_text: str) -> str:
    topics = ", ".join(repo.get("topics") or [])
    parts = [
        f"{repo['full_name']}. {repo.get('description') or ''}".strip(),
        f"Topics: {topics}. Language: {repo.get('language') or 'unknown'}.",
    ]
    if readme_text:
        parts.append(readme_text)
    return "\n".join(p for p in parts if p)


def _doc_hash(doc: str) -> str:
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]


def embed_repos(stars_path: Path, ttl_days: int = 30) -> tuple[list[dict], np.ndarray]:
    repos = json.loads(stars_path.read_text())
    readmes = enrich_readmes(stars_path, ttl_days=ttl_days)

    docs = {}
    hashes = {}
    for r in repos:
        key = f"{r['owner']}/{r['name']}"
        doc = build_doc(r, readmes.get(key, ""))
        docs[str(r["id"])] = doc
        hashes[str(r["id"])] = _doc_hash(doc)

    old_meta: dict = {}
    old_matrix: np.ndarray | None = None
    if META_PATH.exists() and EMBEDDINGS_PATH.exists():
        old_meta = json.loads(META_PATH.read_text())
        old_matrix = np.load(EMBEDDINGS_PATH)

    old_index = {rid: i for i, rid in enumerate(old_meta.get("order", []))}
    old_hashes = old_meta.get("hashes", {})

    to_encode_ids = [
        rid
        for rid in hashes
        if not (rid in old_index and old_hashes.get(rid) == hashes[rid])
    ]

    print(
        f"[embed] {len(to_encode_ids)}/{len(repos)} repos need (re-)embedding "
        f"({len(repos) - len(to_encode_ids)} reused from cache)"
    )

    new_vectors: dict[str, np.ndarray] = {}
    if to_encode_ids:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        texts = [docs[rid] for rid in to_encode_ids]
        vecs = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for rid, vec in zip(to_encode_ids, vecs):
            new_vectors[rid] = vec

    dim = (
        new_vectors[to_encode_ids[0]].shape[0]
        if new_vectors
        else (old_matrix.shape[1] if old_matrix is not None else 384)
    )

    order = [str(r["id"]) for r in repos]
    matrix = np.zeros((len(order), dim), dtype=np.float32)
    for i, rid in enumerate(order):
        if rid in new_vectors:
            matrix[i] = new_vectors[rid]
        elif rid in old_index:
            matrix[i] = old_matrix[old_index[rid]]
        else:
            raise RuntimeError(f"missing embedding for repo id {rid}")

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)
    META_PATH.write_text(json.dumps({"order": order, "hashes": hashes}, indent=2))

    print(f"[embed] wrote {matrix.shape} -> {EMBEDDINGS_PATH}")
    return repos, matrix
