set shell := ["bash", "-uc"]

login := env_var_or_default("GITHUB_LOGIN", "doraeminemon")

# Show available recipes
default:
    @just --list

# Fetch the full starred repo list from GitHub into data/stars.json
fetch:
    uv run stars fetch --login {{login}}

# Fetch/refresh README cache for all starred repos (TTL-aware)
enrich ttl_days="30":
    uv run stars enrich --ttl-days {{ttl_days}}

# Compute sentence embeddings for all repos
embed:
    uv run stars embed

# Run BERTopic clustering + stable slug registry update
topics min_cluster_size="8":
    uv run stars topics --min-cluster-size {{min_cluster_size}}

# Export site/src/data/*.json and search assets
export:
    uv run stars export

# Full data pipeline: fetch -> enrich -> embed -> topics -> export
all: fetch enrich embed topics export

# Fetch the quantized ONNX model used by the browser search island
onnx:
    uv run python scripts/fetch_onnx.py

# Astro dev server
dev:
    cd site && npm run dev

# Astro production build
build:
    cd site && npm run build

# Serve the built site locally
serve:
    cd site && npx serve dist
