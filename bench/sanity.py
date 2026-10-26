"""Print a few real answers so you can check the model still works after quantizing.

    python -m bench.sanity --config vllm-awq

Run this before trusting any speed number. A broken model can be very fast.
"""
from __future__ import annotations

import argparse

from .adapters import make_adapter
from .config import DEFAULT_MATRIX, load_matrix

PROMPTS = [
    "Explain in two sentences why decoding in a large language model is memory-bandwidth bound.",
    "What is 17 * 23? Show your work briefly.",
    "Write a Python function that returns the n-th Fibonacci number.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    matrix = load_matrix(args.matrix)
    cfg = matrix.get(args.config)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    adapter = make_adapter(cfg, matrix.settings)
    adapter.load()
    try:
        adapter.prepare(1)
        for p in PROMPTS:
            chat = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
            ids = tok(chat, add_special_tokens=False).input_ids
            r = adapter.generate([ids], args.tokens, keep_text=True)[0]
            print(f"\n### {p}\n{r.text}\n[{r.n_tokens} tokens, ttft {r.ttft_s*1000:.0f} ms, "
                  f"{(r.n_tokens-1)/(r.e2e_s-r.ttft_s):.1f} tok/s]")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
