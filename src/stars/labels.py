"""Human-readable topic labeling.

Default (no API key required): title-case the top keywords from BERTopic's
MMR representation, e.g. "Llm Agent Frameworks".

Optional upgrade: if ANTHROPIC_API_KEY is set, ask claude-sonnet-5 for a
tighter 2-4 word label given the keywords and a few representative repo
descriptions. This is called from topics.py only when a topic's keyword set
has actually changed (or it's brand new), so it stays cheap and labels stay
stable across runs. Falls back silently to the keyword label on any error —
CI must not break without an API key.
"""

from __future__ import annotations

import os


def _keyword_label(keywords: list[str]) -> str:
    """Pick up to 2 non-overlapping keyword phrases, most-relevant first.
    BERTopic's keyword list is heavy on near-duplicate n-grams sharing a
    root word (e.g. "rust", "written rust", "built rust") — without
    dedup this produces labels like "Rust Written Rust Built Rust"."""
    chosen: list[str] = []
    used_words: set[str] = set()
    for kw in keywords:
        if kw.isdigit() or len(kw) <= 1:
            continue
        words = set(kw.split())
        if words & used_words:
            continue
        chosen.append(kw)
        used_words |= words
        if len(chosen) == 2:
            break
    if not chosen:
        chosen = ["misc"]
    return " · ".join(w.title() for w in chosen)


def label_topic(keywords: list[str], sample_docs: list[str]) -> str:
    default = _keyword_label(keywords)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return default

    try:
        return _label_with_claude(keywords, sample_docs) or default
    except Exception as e:  # noqa: BLE001 - labeling must never break the pipeline
        print(f"[labels] Claude labeling failed, falling back to keywords: {e}")
        return default


def _label_with_claude(keywords: list[str], sample_docs: list[str]) -> str | None:
    import anthropic

    client = anthropic.Anthropic()
    sample = "\n".join(f"- {d[:160]}" for d in sample_docs[:10])
    prompt = (
        "You are naming a topic cluster of GitHub repositories for a browsing UI.\n"
        f"Top keywords for this cluster: {', '.join(keywords[:10])}\n"
        f"Representative repos:\n{sample}\n\n"
        "Reply with ONLY a concise 2-4 word title-case label for this cluster "
        "(e.g. 'LLM Agent Frameworks', 'Rust CLI Tools'). No punctuation, no quotes, "
        "no explanation."
    )
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()
    return text or None
