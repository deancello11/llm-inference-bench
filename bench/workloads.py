"""Makes the prompts used in the speed tests.

Every stack gets token ids instead of text, so the prompt length is exactly the same everywhere.
The tokens are random and different for every request. That way prefix caching can't make
the prompt processing look free, and the content doesn't matter for speed or energy.
"""
from __future__ import annotations

import zlib

import numpy as np

# Token ids in this range are ordinary words and pieces of words, not special tokens
# (the Qwen2.5 vocabulary has about 151k tokens).
TOKEN_LO, TOKEN_HI = 1000, 50000


def seed_for(*parts) -> int:
    """Turn labels like ("short", 8, round, repeat) into a repeatable random seed."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def make_prompts(input_len: int, n: int, seed: int) -> list[list[int]]:
    """n different prompts, each exactly input_len tokens long."""
    rng = np.random.default_rng(seed)
    return rng.integers(TOKEN_LO, TOKEN_HI, size=(n, input_len)).tolist()
