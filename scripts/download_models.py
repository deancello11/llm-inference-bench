"""Pre-download every checkpoint so the benchmark never measures a download.

    python scripts/download_models.py [--only hf-fp16,vllm-awq]

Put HF_HOME on persistent storage so a pod restart doesn't re-download.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download  # noqa: E402

from bench.adapters.llamacpp import gguf_shards  # noqa: E402
from bench.config import load_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    for c in load_matrix().enabled():
        if args.only and c.id not in args.only.split(","):
            continue
        print(f"-> {c.id}: {c.checkpoint} {c.file or ''}")
        if c.file:
            gguf_shards(c)
        else:
            snapshot_download(c.checkpoint, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])


if __name__ == "__main__":
    main()
