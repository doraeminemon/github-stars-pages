"""CLI entrypoints for the stars pipeline (invoked by the justfile)."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

STARS_PATH = Path("data/stars.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
EMBEDDINGS_META_PATH = Path("data/embeddings_meta.json")
TOPICS_RESULT_PATH = Path("data/topics_result.json")


@app.command()
def fetch(login: str = typer.Option(..., help="GitHub username to fetch stars for")):
    """Fetch the full starred-repo list into data/stars.json."""
    from .fetch import fetch_stars

    fetch_stars(login, STARS_PATH)


@app.command()
def enrich(ttl_days: int = typer.Option(30, help="Cache freshness window in days")):
    """Fetch/refresh the README cache for all starred repos."""
    from .readmes import enrich_readmes

    enrich_readmes(STARS_PATH, ttl_days=ttl_days)


@app.command()
def embed(ttl_days: int = typer.Option(30, help="README cache TTL, in case enrich wasn't run separately")):
    """Compute sentence embeddings for all repos (also ensures READMEs are cached)."""
    from .embed import embed_repos

    embed_repos(STARS_PATH, ttl_days=ttl_days)


@app.command()
def topics(min_cluster_size: int = typer.Option(8, help="HDBSCAN min_cluster_size")):
    """Run BERTopic clustering and update the stable slug registry."""
    from .topics import run_topics

    run_topics(STARS_PATH, EMBEDDINGS_PATH, EMBEDDINGS_META_PATH, min_cluster_size=min_cluster_size)


@app.command()
def export():
    """Write site/src/data/*.json and site/public/search/* from pipeline outputs."""
    from .export import export_site_data

    export_site_data(STARS_PATH, EMBEDDINGS_PATH, EMBEDDINGS_META_PATH, TOPICS_RESULT_PATH)


@app.command(name="all")
def run_all(
    login: str = typer.Option(..., help="GitHub username to fetch stars for"),
    ttl_days: int = typer.Option(30, help="README cache TTL in days"),
    min_cluster_size: int = typer.Option(8, help="HDBSCAN min_cluster_size"),
):
    """Run the full pipeline: fetch -> enrich -> embed -> topics -> export."""
    from .embed import embed_repos
    from .export import export_site_data
    from .fetch import fetch_stars
    from .readmes import enrich_readmes
    from .topics import run_topics

    fetch_stars(login, STARS_PATH)
    enrich_readmes(STARS_PATH, ttl_days=ttl_days)
    embed_repos(STARS_PATH, ttl_days=ttl_days)
    run_topics(STARS_PATH, EMBEDDINGS_PATH, EMBEDDINGS_META_PATH, min_cluster_size=min_cluster_size)
    export_site_data(STARS_PATH, EMBEDDINGS_PATH, EMBEDDINGS_META_PATH, TOPICS_RESULT_PATH)


if __name__ == "__main__":
    app()
