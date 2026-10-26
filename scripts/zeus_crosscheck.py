"""Compares our energy numbers with the Zeus library (from the ML.ENERGY project).

    python scripts/zeus_crosscheck.py --config hf-fp16 --tokens 256 --batch 1

It generates text once while both our GpuMonitor and Zeus measure the energy, then prints
how far apart they are. They should be within a few percent. A big gap means one of them
has a bug. Needs `pip install zeus` in the Hugging Face environment (older releases call it zeus-ml).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.adapters import make_adapter  # noqa: E402
from bench.config import load_matrix  # noqa: E402
from bench.gpu import GpuMonitor  # noqa: E402
from bench.workloads import make_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hf-fp16")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    from zeus.monitor import ZeusMonitor

    matrix = load_matrix()
    adapter = make_adapter(matrix.get(args.config), matrix.settings)
    adapter.load()
    adapter.prepare(args.batch)
    adapter.generate(make_prompts(128, args.batch, 1), 32)  # warmup

    mon = GpuMonitor(0)
    mon.start()
    zeus = ZeusMonitor(gpu_indices=[0], sync_execution_with="torch")
    zeus.begin_window("run")
    w = mon.begin_window()
    for i in range(args.iters):
        adapter.generate(make_prompts(128, args.batch, 100 + i), args.tokens)
    ours = mon.end_window(w)
    z = zeus.end_window("run")
    mon.stop()

    toks = args.iters * args.batch * args.tokens
    print(f"ours : {ours['energy_j']:.2f} J  ({1000*ours['energy_j']/toks:.2f} mJ/token) over {ours['duration_s']:.2f}s")
    print(f"zeus : {z.total_energy:.2f} J  ({1000*z.total_energy/toks:.2f} mJ/token) over {z.time:.2f}s")
    print(f"relative difference: {abs(ours['energy_j']-z.total_energy)/z.total_energy*100:.2f}%")
    adapter.close()


if __name__ == "__main__":
    main()
