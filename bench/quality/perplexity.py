"""WikiText-2 perplexity with an identical protocol across HF and vLLM.

Protocol: tokenize the raw test split joined with "\\n\\n", cut into NON-overlapping
windows of `ctx` tokens (default 2048), drop the incomplete tail, score tokens 1..ctx-1 of
each window. (Non-overlapping => a bit higher PPL than a strided sliding window, but the
same for every config.)

llama.cpp is scored with its own `llama-perplexity` binary, whose chunking/scoring differs
(it scores only the second half of each window by default). So llama.cpp numbers are only
comparable *within llama.cpp* (use llamacpp-f16 as that stack's reference).
"""
from __future__ import annotations

import math
import re
import subprocess

from ..adapters.llamacpp import find_binary, gguf_path
from ..config import ROOT


def wikitext2_text() -> str:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def make_windows(tokenizer, ctx: int = 2048, max_windows: int | None = None) -> list[list[int]]:
    ids = tokenizer(wikitext2_text(), add_special_tokens=False).input_ids
    windows = [ids[i:i + ctx] for i in range(0, len(ids) - ctx + 1, ctx)]
    return windows[:max_windows] if max_windows else windows


def run_ppl(adapter, tokenizer, ctx: int = 2048, max_windows: int | None = None) -> dict:
    windows = make_windows(tokenizer, ctx, max_windows)
    scored = adapter.score_windows(windows)
    nll = sum(s[0] for s in scored)
    n = sum(s[1] for s in scored)
    return {"ppl": math.exp(nll / n), "ppl_tokens": n, "ppl_windows": len(windows), "ppl_ctx": ctx,
            "ppl_protocol": "nonoverlap-full-window"}


def run_llamacpp_ppl(cfg, ctx: int = 2048, max_chunks: int | None = None) -> dict:
    binary = find_binary("llama-perplexity", "LLAMA_PERPLEXITY_BIN")
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    txt = data / "wikitext2_test.txt"
    if not txt.exists():
        txt.write_text(wikitext2_text(), encoding="utf-8")
    cmd = [binary, "-m", gguf_path(cfg), "-f", str(txt), "-c", str(ctx), "-ngl", "99"]
    if max_chunks:
        cmd += ["--chunks", str(max_chunks)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(r"Final estimate: PPL = ([\d.]+)", text)
    if not m:
        raise RuntimeError("could not parse llama-perplexity output:\n" + text[-2000:])
    return {"ppl": float(m.group(1)), "ppl_ctx": ctx, "ppl_protocol": "llama-perplexity (second-half scoring)"}
