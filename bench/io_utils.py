"""Helpers for saving and reading results (one JSON record per line)."""
from __future__ import annotations

import json
import os
import pathlib
import time

from .config import RESULTS_DIR


def results_path(name: str, out_dir: pathlib.Path | None = None) -> pathlib.Path:
    d = pathlib.Path(out_dir) if out_dir else RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    """Add one record to the end of the file and force it to disk.

    If the pod crashes, every finished run is already saved.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # the last line can be cut off if the process was killed mid-write
    return out


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_key(r: dict) -> tuple:
    """Identifies one measurement, so a re-run can skip the ones already done."""
    return (r["config_id"], r["workload"], r["concurrency"], r["round"], r["repeat"])


def checkpoint_bytes(repo: str) -> int:
    """Size of the weight files in a Hugging Face repo (what the GPU reads for every token)."""
    from huggingface_hub import snapshot_download

    d = pathlib.Path(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json", "*.txt"]))
    files = list(d.glob("*.safetensors")) or list(d.glob("*.bin"))
    return sum(f.stat().st_size for f in files)
