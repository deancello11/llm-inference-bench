"""Quality evals for one config.

    python -m bench.quality.run_quality --config vllm-awq --tasks ppl,arc,mmlu,hellaswag
    python -m bench.quality.run_quality --config hf-fp16 --tasks arc --limit 50      # smoke test

Results are appended to results/quality.jsonl (one row per config x task).
"""
from __future__ import annotations

import argparse
import time

from ..adapters import make_adapter
from ..config import DEFAULT_MATRIX, load_matrix
from ..io_utils import append_jsonl, now_iso, results_path
from . import mc_eval, perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", default="ppl,arc,mmlu,hellaswag")
    ap.add_argument("--limit", type=int, default=None, help="cap items per task (smoke tests)")
    ap.add_argument("--ppl-ctx", type=int, default=2048)
    ap.add_argument("--determinism-check", action="store_true", help="score the first 100 ARC items twice")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    cfg = matrix.get(args.config)
    out = results_path("quality.jsonl", args.out_dir)
    tasks = args.tasks.split(",")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)

    def emit(task, payload):
        append_jsonl(out, {"ts": now_iso(), "config_id": cfg.id, "stack": cfg.stack, "precision": cfg.precision,
                           "task": task, **payload})
        print(f"[{cfg.id}] {task}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in payload.items()))

    # llama.cpp perplexity uses its own binary and must run before the server holds the GPU.
    if "ppl" in tasks and cfg.stack == "llamacpp":
        emit("ppl", perplexity.run_llamacpp_ppl(cfg, args.ppl_ctx))
        tasks = [t for t in tasks if t != "ppl"]

    adapter = make_adapter(cfg, matrix.settings)
    adapter.load()
    try:
        adapter.prepare(1)
        for task in tasks:
            t0 = time.perf_counter()
            if task == "ppl":
                emit("ppl", perplexity.run_ppl(adapter, tok, args.ppl_ctx, max_windows=args.limit))
            else:
                items = mc_eval.build_items(task, limit=args.limit)
                r = mc_eval.evaluate(adapter, tok, items)
                r.pop("preds")
                emit(task, {**r, "eval_s": time.perf_counter() - t0})
        if args.determinism_check:
            items = mc_eval.build_items("arc", limit=100)
            a = mc_eval.evaluate(adapter, tok, items)["preds"]
            b = mc_eval.evaluate(adapter, tok, items)["preds"]
            emit("determinism_arc100", {"match_frac": sum(x == y for x, y in zip(a, b)) / len(a)})
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
