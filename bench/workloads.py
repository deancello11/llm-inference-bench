"""Deterministic prompt generation.

Every stack receives *token ids* (not text), so input length is exactly the same
everywhere regardless of tokenizer round-tripping. Prompts are random tokens from a
common-vocabulary range: for throughput/latency/energy the content doesn't matter,
but it must be *different per request* so prefix caching (vLLM, llama.cpp) can't turn
prefill into a cache hit and make a stack look artificially fast.
"""
from __future__ import annotations

import zlib

import numpy as np

# Qwen2.5 vocab is ~151k; ids in this range are ordinary text pieces, not special tokens.
TOKEN_LO, TOKEN_HI = 1000, 50000


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def make_prompts(input_len: int, n: int, seed: int) -> list[list[int]]:
    """n distinct prompts of exactly input_len token ids."""
    rng = np.random.default_rng(seed)
    return rng.integers(TOKEN_LO, TOKEN_HI, size=(n, input_len)).tolist()
