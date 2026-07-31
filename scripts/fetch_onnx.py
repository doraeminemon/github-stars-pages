"""Fetch the ONNX export of all-MiniLM-L6-v2 for the browser search island.

Downloaded into site/public/models/Xenova/all-MiniLM-L6-v2/ at build time
(gitignored, not committed) — @huggingface/transformers is configured with
env.allowRemoteModels = false and env.localModelPath pointing here, so the
deployed site makes zero third-party requests at runtime.

Only fetches model_quantized.onnx (~23MB): search.ts calls the pipeline with
dtype: "q8", which @huggingface/transformers maps to the "_quantized" file
suffix (see DATA_TYPES in its source). public/ is copied byte-for-byte into
the Pages deploy, so pulling every precision variant here (fp32, fp16,
int8, ...) would needlessly balloon the deployed artifact to ~10x the size
actually served to visitors.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Xenova/all-MiniLM-L6-v2"
TARGET_DIR = Path("site/public/models") / REPO_ID


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        allow_patterns=["*.json", "*.txt", "onnx/model_quantized.onnx"],
    )
    total_bytes = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"[fetch_onnx] {REPO_ID} -> {path} ({total_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
