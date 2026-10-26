"""Makes fake results and checks that the chart code runs from start to finish."""
import json
import random

from analysis.make_charts import main
from bench.config import load_matrix

# Made-up numbers for each config:
# (decode tok/s at c=1, weight GB, accuracy, perplexity, mJ/token at c=1)
FAKE = {
    "hf-fp16": (60, 6.2, 0.70, 8.1, 900), "hf-int8-bnb": (20, 3.5, 0.69, 8.2, 2500),
    "hf-nf4-bnb": (45, 2.2, 0.66, 8.8, 1100), "vllm-fp16": (140, 6.2, 0.70, 8.1, 400),
    "vllm-awq": (250, 2.1, 0.68, 8.5, 230), "vllm-gptq": (240, 2.1, 0.67, 8.6, 240),
    "llamacpp-f16": (120, 6.2, 0.70, 8.1, 450), "llamacpp-q8": (190, 3.3, 0.70, 8.1, 300),
    "llamacpp-q4km": (260, 1.9, 0.67, 8.5, 220),
}


def synth(tmp):
    m = load_matrix()
    rng = random.Random(0)
    runs, loads, qual = [], [], []
    for c in m.enabled():
        tps, gb, acc, ppl, mj = FAKE[c.id]
        loads.append({"config_id": c.id, "weights_bytes": gb * 1e9, "idle_power_w": 25.0, "kv_cache_tokens": 100000})
        for t, v in (("arc", acc), ("mmlu", acc - .05), ("hellaswag", acc + .05)):
            qual.append({"config_id": c.id, "stack": c.stack, "precision": c.precision, "task": t, "acc": v, "n": 1000})
        qual.append({"config_id": c.id, "stack": c.stack, "precision": c.precision, "task": "ppl", "ppl": ppl})
        for wn, wl in m.workloads.items():
            for conc in m.concurrency:
                for rep in range(3):
                    n = rng.uniform(0.97, 1.03)
                    agg = tps * conc ** 0.8 * n
                    tokens = 256 * conc * 5
                    dur = tokens / agg
                    e = mj * conc ** -0.7 * tokens / 1000
                    runs.append({
                        "config_id": c.id, "stack": c.stack, "precision": c.precision, "workload": wn,
                        "input_len": wl.input_len, "concurrency": conc, "round": 0, "repeat": rep,
                        "total_tokens": tokens, "ttft_mean_s": 0.02 * wl.input_len / 128 * n, "ttft_p95_s": 0.03,
                        "decode_tps_per_request": tps * conc ** -0.1 * n, "itl_mean_ms": 1000 / tps,
                        "throughput_tps": agg, "energy_per_token_mj": 1000 * e / tokens,
                        "energy_idle_subtracted_j": e * 0.8, "duration_s": dur, "energy_j": e,
                        "mem_delta_peak_mib": 7000, "power_mean_w": 300, "sm_clock_mean_mhz": 2500,
                        "temp_max_c": 70, "thermal_throttled": False,
                    })
    for name, rows in (("runs", runs), ("loads", loads), ("quality", qual)):
        (tmp / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_charts_render(tmp_path, monkeypatch):
    synth(tmp_path)
    out = tmp_path / "charts"
    monkeypatch.setattr("sys.argv", ["x", "--results", str(tmp_path), "--out", str(out)])
    main()
    for f in ("pareto.png", "throughput_vs_concurrency.png", "energy_vs_concurrency.png", "ttft_vs_input.png",
              "memory_weights.png", "roofline.png", "summary.csv", "summary_short_c1.md"):
        assert (out / f).exists(), f
