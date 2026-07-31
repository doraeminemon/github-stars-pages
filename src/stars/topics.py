"""BERTopic clustering over precomputed embeddings, with a stable slug
registry so that /topic/<slug> URLs don't shuffle every time new stars are
added and the clustering is recomputed.

Topic *identity* is tracked by comparing each new topic's centroid (mean of
its member embeddings, in the original 384-dim space) against the centroids
recorded in data/topic_registry.json from the previous run. A cosine
similarity >= REGISTRY_MATCH_THRESHOLD reuses the old slug (and label, if the
topic's keyword set hasn't changed); anything unmatched gets a fresh slug.
data/topic_registry.json is the one generated file that gets committed to
git, specifically so this history persists across CI runs.

Parent groupings for homepage navigation are computed with agglomerative
clustering directly over topic centroids rather than BERTopic's raw
hierarchical linkage output — same purpose (coarse groups of related
topics), simpler and more robust to parse/maintain.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .labels import label_topic

REGISTRY_PATH = Path("data/topic_registry.json")
RESULT_PATH = Path("data/topics_result.json")
REGISTRY_MATCH_THRESHOLD = 0.75
OUTLIER_TOPIC_ID = -1


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s) or "topic"


def _unique_slug(base: str, taken: set[str]) -> str:
    slug = base
    n = 2
    while slug in taken:
        slug = f"{base}-{n}"
        n += 1
    taken.add(slug)
    return slug


def _keywords_hash(keywords: list[str]) -> str:
    return hashlib.sha256(",".join(keywords).encode()).hexdigest()[:12]


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return an @ bn.T


def _build_topic_model(min_cluster_size: int):
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=True,
    ), vectorizer_model, ctfidf_model, representation_model


def _parent_groups(topic_ids: list[int], centroids: np.ndarray) -> dict[int, int]:
    """Coarse grouping of topics for homepage navigation. Returns
    {topic_id: parent_group_index}."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(topic_ids)
    if n <= 1:
        return {tid: 0 for tid in topic_ids}
    n_groups = max(2, min(12, round(n / 6)))
    n_groups = min(n_groups, n)
    # ward/euclidean gives much more balanced group sizes than
    # cosine/average here: topic centroids from short software-repo
    # descriptions sit close together in cosine terms, so average-linkage
    # chains nearly everything into one dominant blob before reaching the
    # target cut (empirically: one group holding 36/46 topics).
    clustering = AgglomerativeClustering(n_clusters=n_groups, metric="euclidean", linkage="ward")
    labels = clustering.fit_predict(centroids)
    return {tid: int(lbl) for tid, lbl in zip(topic_ids, labels)}


def run_topics(
    stars_path: Path,
    embeddings_path: Path,
    embeddings_meta_path: Path,
    min_cluster_size: int = 12,
) -> dict:
    repos = json.loads(stars_path.read_text())
    matrix = np.load(embeddings_path)
    meta = json.loads(embeddings_meta_path.read_text())
    order = meta["order"]  # list of repo id strings, aligned with matrix rows

    id_to_repo = {str(r["id"]): r for r in repos}
    docs = []
    for rid in order:
        r = id_to_repo[rid]
        docs.append(f"{r['full_name']}. {r.get('description') or ''}")

    topic_model, vectorizer_model, ctfidf_model, representation_model = _build_topic_model(
        min_cluster_size
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=matrix)

    if OUTLIER_TOPIC_ID in topics:
        reduced = topic_model.reduce_outliers(
            docs, topics, strategy="embeddings", embeddings=matrix
        )
        # IMPORTANT: update_topics() silently resets vectorizer_model/ctfidf_model/
        # representation_model to bare defaults (plain CountVectorizer with NO
        # stopword filtering) unless they're explicitly re-passed here. Without
        # this, topic keywords fill up with "for", "and", "with", etc.
        topic_model.update_topics(
            docs,
            topics=reduced,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
        )
    else:
        reduced = topics

    unique_topic_ids = sorted(set(reduced) - {OUTLIER_TOPIC_ID})

    # Per-topic keywords, member indices, centroid.
    topic_keywords: dict[int, list[str]] = {}
    topic_members: dict[int, list[int]] = {tid: [] for tid in unique_topic_ids}
    for idx, tid in enumerate(reduced):
        if tid != OUTLIER_TOPIC_ID:
            topic_members[tid].append(idx)

    for tid in unique_topic_ids:
        words = topic_model.get_topic(tid) or []
        topic_keywords[tid] = [w for w, _ in words[:10]] or [f"topic-{tid}"]

    centroids = {
        tid: matrix[topic_members[tid]].mean(axis=0) for tid in unique_topic_ids
    }
    centroid_matrix = np.stack([centroids[tid] for tid in unique_topic_ids]) if unique_topic_ids else np.zeros((0, matrix.shape[1]))

    parent_map = _parent_groups(unique_topic_ids, centroid_matrix) if unique_topic_ids else {}

    # --- Registry matching for stable slugs -------------------------------
    old_registry: list[dict] = []
    if REGISTRY_PATH.exists():
        old_registry = json.loads(REGISTRY_PATH.read_text())

    old_centroids = (
        np.array([e["centroid"] for e in old_registry]) if old_registry else np.zeros((0, matrix.shape[1]))
    )

    sims = (
        _cosine_sim_matrix(centroid_matrix, old_centroids)
        if len(unique_topic_ids) and len(old_registry)
        else np.zeros((len(unique_topic_ids), 0))
    )

    # Greedy best-first matching across all (new, old) pairs.
    pairs = []
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            if sims[i, j] >= REGISTRY_MATCH_THRESHOLD:
                pairs.append((sims[i, j], i, j))
    pairs.sort(reverse=True)

    matched_new: dict[int, int] = {}  # position in unique_topic_ids -> old_registry index
    used_old: set[int] = set()
    for _, i, j in pairs:
        if i in matched_new or j in used_old:
            continue
        matched_new[i] = j
        used_old.add(j)

    taken_slugs = {e["slug"] for e in old_registry}
    new_registry: list[dict] = []
    topic_records: list[dict] = []

    for pos, tid in enumerate(unique_topic_ids):
        keywords = topic_keywords[tid]
        khash = _keywords_hash(keywords)
        size = len(topic_members[tid])
        centroid = centroids[tid]

        if pos in matched_new:
            old_entry = old_registry[matched_new[pos]]
            slug = old_entry["slug"]
            if old_entry.get("keywords_hash") == khash and old_entry.get("label"):
                label = old_entry["label"]
            else:
                label = label_topic(keywords, [docs[i] for i in topic_members[tid][:10]])
        else:
            label = label_topic(keywords, [docs[i] for i in topic_members[tid][:10]])
            slug = _unique_slug(_slugify(label), taken_slugs)

        new_registry.append(
            {
                "slug": slug,
                "centroid": centroid.tolist(),
                "label": label,
                "keywords": keywords,
                "keywords_hash": khash,
                "size": size,
            }
        )
        topic_records.append(
            {
                "slug": slug,
                "label": label,
                "keywords": keywords,
                "size": size,
                "parent_group": parent_map.get(tid, 0),
                "repo_ids": [order[i] for i in topic_members[tid]],
            }
        )

    tid_to_pos = {tid: pos for pos, tid in enumerate(unique_topic_ids)}
    assignments = {}
    for idx, tid in enumerate(reduced):
        rid = order[idx]
        if tid == OUTLIER_TOPIC_ID:
            assignments[rid] = None
        else:
            assignments[rid] = topic_records[tid_to_pos[tid]]["slug"]

    n_outliers = sum(1 for v in assignments.values() if v is None)
    print(
        f"[topics] {len(unique_topic_ids)} topics over {len(order)} repos "
        f"({n_outliers} unassigned, {n_outliers / len(order):.1%})"
    )

    REGISTRY_PATH.write_text(json.dumps(new_registry, indent=2))

    result = {"topics": topic_records, "assignments": assignments}
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[topics] wrote {RESULT_PATH}")
    return result
