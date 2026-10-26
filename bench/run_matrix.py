"""Runs every config, workload and concurrency level, with no one watching.

    python -m bench.run_matrix --rounds 3 --repeats-per-load 2 --out-dir results

Repeats per cell = rounds x repeats-per-load. In each round every config is run once, in a
new random order. The repeats are spread across rounds so that slow changes (a warming GPU,
a busy neighbor on the machine) affect all stacks equally and don't favor whichever ran last.

Each stack can use its own virtualenv, because vLLM needs its own version of torch. Point to
them with the BENCH_PY_HF, BENCH_PY_VLLM and BENCH_PY_LLAMACPP environment variables.

It is safe to stop this at any time and start it again. Finished runs are skipped.
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
    """The Python interpreter to use for a stack (its own venv if one is set)."""
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
            # We call the venv's python directly, so add its bin folder to PATH.
            # vLLM needs the `ninja` tool from there to build a small GPU kernel.
            env = {**os.environ, "PATH": os.path.dirname(cmd[0]) + os.pathsep + os.environ.get("PATH", "")}
            rc = subprocess.run(cmd, env=env).returncode
            if rc != 0:  # a failed config is saved as a finding and we keep going
                append_jsonl(fails_f, {"ts": now_iso(), "config_id": cid, "round": rnd,
                                       "stage": "process", "error": f"exit code {rc}"})
                print(f"!! {cid} round {rnd} exited with {rc} (logged; continuing)")
            time.sleep(args.gpu_cooldown_s)  # let the GPU rest and free its memory before the next config


if __name__ == "__main__":
    main()
