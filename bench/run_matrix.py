"""Run the whole grid unattended, resumable, with interleaved ordering.

    python -m bench.run_matrix --rounds 3 --repeats-per-load 2 --out-dir results

Total repeats per cell = rounds x repeats-per-load. Each round visits every config in a
fresh random order, loading each model once and measuring `repeats-per-load` windows per
cell. Spreading repeats over rounds means slow thermal drift or a noisy neighbour hits all
stacks equally instead of biasing whichever ran last.

Each stack can use its own virtualenv (vLLM pins its own torch). Point at them with
BENCH_PY_HF, BENCH_PY_VLLM, BENCH_PY_LLAMACPP (default: this interpreter).
Safe to Ctrl-C / kill the pod and re-run: finished windows are skipped.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

from .config import DEFAULT_MATRIX, load_matrix
from .io_utils import append_jsonl, now_iso, results_path


def stack_python(stack: str) -> str:
    return os.environ.get(f"BENCH_PY_{stack.upper()}", sys.executable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats-per-load", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma list of config ids")
    ap.add_argument("--skip", default="", help="comma list of config ids")
    ap.add_argument("--min-window-s", type=float, default=10.0)
    ap.add_argument("--gpu-cooldown-s", type=int, default=30, help="pause between loads")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    ids = [c.id for c in matrix.enabled()]
    if args.only:
        ids = [i for i in ids if i in args.only.split(",")]
    ids = [i for i in ids if i not in args.skip.split(",")]
    fails_f = results_path("failures.jsonl", args.out_dir)

    for rnd in range(args.rounds):
        order = ids[:]
        random.Random(rnd).shuffle(order)
        print(f"=== round {rnd}: {order}")
        for cid in order:
            cfg = matrix.get(cid)
            cmd = [stack_python(cfg.stack), "-m", "bench.run_config", "--matrix", args.matrix,
                   "--config", cid, "--round", str(rnd), "--repeats", str(args.repeats_per_load),
                   "--min-window-s", str(args.min_window_s)]
            if args.out_dir:
                cmd += ["--out-dir", args.out_dir]
            print("$", " ".join(cmd))
            if args.dry_run:
                continue
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                append_jsonl(fails_f, {"ts": now_iso(), "config_id": cid, "round": rnd,
                                       "stage": "process", "error": f"exit code {rc}"})
                print(f"!! {cid} round {rnd} exited with {rc} (logged; continuing)")
            time.sleep(args.gpu_cooldown_s)


if __name__ == "__main__":
    main()
