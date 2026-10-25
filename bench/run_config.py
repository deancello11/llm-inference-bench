"""Measure ONE config across workloads x concurrency x repeats.

    python -m bench.run_config --config hf-fp16 --repeats 3

Each repeat is one "measurement window": the generate() call is run back-to-back until
the window lasts at least --min-window-s (default 10 s) so the energy reading is stable,
and latency stats are pooled over every request in the window. One JSON line is written
per window, with the config attached. Finished windows are skipped on re-run (resumable).
"""
from __future__ import annotations

import argparse
import random
import time
import traceback

from .adapters import make_adapter
from .config import DEFAULT_MATRIX, load_matrix
from .env_info import collect_env
from .gpu import GpuMonitor
from .io_utils import append_jsonl, now_iso, read_jsonl, results_path, run_key
from .stats import mean, percentile
from .workloads import make_prompts, seed_for


def measure_window(adapter, monitor, wl, conc, round_idx, repeat, min_window_s, max_iters=50):
    """One timed window: generate (possibly several times) while recording energy/clocks."""
    adapter.reset_peak()
    results = []
    w = monitor.begin_window()
    t_start = time.perf_counter()
    it = 0
    while True:
        prompts = make_prompts(wl.input_len, conc, seed_for(wl.name, conc, round_idx, repeat, it))
        results.extend(adapter.generate(prompts, wl.output_len))
        it += 1
        if time.perf_counter() - t_start >= min_window_s or it >= max_iters:
            break
    g = monitor.end_window(w)

    bad = [r.n_tokens for r in results if r.n_tokens != wl.output_len]
    if bad:
        raise RuntimeError(f"length mismatch: expected {wl.output_len} tokens, got {sorted(set(bad))}")

    total_tokens = sum(r.n_tokens for r in results)
    ttfts = [r.ttft_s for r in results]
    decode_tps = [(r.n_tokens - 1) / (r.e2e_s - r.ttft_s) for r in results if r.e2e_s > r.ttft_s]
    return {
        "n_iters": it,
        "total_tokens": total_tokens,
        "ttft_mean_s": mean(ttfts),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "e2e_mean_s": mean([r.e2e_s for r in results]),
        "decode_tps_per_request": mean(decode_tps),           # 1 / inter-token latency
        "itl_mean_ms": 1000.0 / mean(decode_tps),
        "throughput_tps": total_tokens / g["duration_s"],       # aggregate, includes prefill
        "energy_per_token_mj": 1000.0 * g["energy_j"] / total_tokens,
        **g,
        **adapter.extra_metrics(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--config", required=True)
    ap.add_argument("--workloads", default=None, help="comma list; default all")
    ap.add_argument("--concurrency", default=None, help="comma list; default from matrix")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--round", type=int, default=0, help="round index (matrix runner interleaves rounds)")
    ap.add_argument("--warmup-iters", type=int, default=1)
    ap.add_argument("--min-window-s", type=float, default=10.0)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    cfg = matrix.get(args.config)
    wl_names = args.workloads.split(",") if args.workloads else list(matrix.workloads)
    concs = [int(x) for x in args.concurrency.split(",")] if args.concurrency else matrix.concurrency

    runs_f = results_path("runs.jsonl", args.out_dir)
    loads_f = results_path("loads.jsonl", args.out_dir)
    fails_f = results_path("failures.jsonl", args.out_dir)
    done = {run_key(r) for r in read_jsonl(runs_f)}

    # cells still to do for this (config, round)
    cells = [(wn, c) for c in concs for wn in wl_names
             if any((cfg.id, wn, c, args.round, r) not in done for r in range(args.repeats))]
    if not cells:
        print(f"[{cfg.id}] round {args.round}: nothing to do")
        return
    random.Random(args.seed * 1000 + args.round).shuffle(cells)
    cells.sort(key=lambda x: x[1])  # group by concurrency (llama.cpp restarts per level); stable within group

    monitor = GpuMonitor(args.gpu_index)
    monitor.start()
    info = monitor.info()
    print(f"[{cfg.id}] GPU: {info}")
    if info.get("mig_enabled"):
        raise SystemExit("MIG is enabled: you do not have a full GPU. Change instance.")
    if not info["energy_counter"]:
        print("WARNING: NVML energy counter unsupported; falling back to integrated power samples.")

    mem_baseline = monitor.mem_used_mib()
    adapter = make_adapter(cfg, matrix.settings)
    t0 = time.perf_counter()
    try:
        adapter.load()
    except Exception as e:  # noqa: BLE001
        append_jsonl(fails_f, {"ts": now_iso(), "config_id": cfg.id, "stage": "load", "error": repr(e),
                               "trace": traceback.format_exc()})
        raise
    load_s = time.perf_counter() - t0

    time.sleep(3)
    idle_w = monitor.measure_idle_power_w(matrix.settings.get("idle_seconds", 20))
    append_jsonl(loads_f, {
        "ts": now_iso(), "config_id": cfg.id, "round": args.round, "load_s": load_s,
        "idle_power_w": idle_w, "mem_baseline_mib": mem_baseline,
        "mem_after_load_mib": monitor.mem_used_mib(), "gpu": info,
        "checkpoint": cfg.checkpoint, "file": cfg.file, **adapter.load_info, "env": collect_env(),
    })
    print(f"[{cfg.id}] loaded in {load_s:.1f}s, idle power {idle_w:.1f} W")

    try:
        for wn, conc in cells:
            wl = matrix.workloads[wn]
            try:
                adapter.prepare(conc)
                for wi in range(args.warmup_iters):  # discarded
                    adapter.generate(make_prompts(wl.input_len, conc, seed_for("warm", wn, conc, wi)), 32)
                for rep in range(args.repeats):
                    if (cfg.id, wn, conc, args.round, rep) in done:
                        continue
                    m = measure_window(adapter, monitor, wl, conc, args.round, rep, args.min_window_s)
                    rec = {
                        "ts": now_iso(), "config_id": cfg.id, "stack": cfg.stack, "precision": cfg.precision,
                        "workload": wn, "input_len": wl.input_len, "output_len": wl.output_len,
                        "concurrency": conc, "round": args.round, "repeat": rep, "idle_power_w": idle_w,
                        "energy_idle_subtracted_j": m["energy_j"] - idle_w * m["duration_s"],
                        "mem_delta_peak_mib": m["mem_used_peak_mib"] - mem_baseline,
                        **m,
                    }
                    append_jsonl(runs_f, rec)
                    print(f"[{cfg.id}] {wn} c={conc} rep={rep}: ttft={m['ttft_mean_s']*1000:.0f}ms "
                          f"decode={m['decode_tps_per_request']:.1f} tok/s/req agg={m['throughput_tps']:.0f} tok/s "
                          f"E={m['energy_per_token_mj']:.1f} mJ/tok sm={m['sm_clock_mean_mhz']:.0f}MHz "
                          f"T={m['temp_max_c']:.0f}C{' THROTTLED' if m['thermal_throttled'] else ''}")
            except Exception as e:  # noqa: BLE001  -- failures are findings; log and continue
                append_jsonl(fails_f, {"ts": now_iso(), "config_id": cfg.id, "workload": wn, "concurrency": conc,
                                       "round": args.round, "stage": "run", "error": repr(e),
                                       "trace": traceback.format_exc()})
                print(f"[{cfg.id}] {wn} c={conc} FAILED: {e!r}")
                if "CUDA error" in str(e):  # context is poisoned; stop this process
                    break
    finally:
        adapter.close()
        monitor.stop()


if __name__ == "__main__":
    main()
