"""The one interface every serving stack implements.

The measurement harness only ever calls `generate()`, so all stacks go through
identical timing/energy logic. Quality evals use `score_windows()` and
`next_token_topk_batch()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import ModelConfig


@dataclass
class GenResult:
    ttft_s: float          # request start -> first generated token available
    e2e_s: float           # request start -> last token
    n_tokens: int          # generated tokens (must equal the requested output length)
    text: str | None = None


class Adapter(ABC):
    def __init__(self, cfg: ModelConfig, settings: dict):
        self.cfg = cfg
        self.settings = settings
        self.load_info: dict = {}

    @abstractmethod
    def load(self) -> None:
        """Load the model onto the GPU. Fill self.load_info (weights_bytes, etc.)."""

    def prepare(self, concurrency: int) -> None:
        """Called before each concurrency level (llama.cpp restarts its server here)."""

    @abstractmethod
    def generate(self, prompts: list[list[int]], max_new_tokens: int, keep_text: bool = False) -> list[GenResult]:
        """Greedy-generate exactly max_new_tokens for each prompt *concurrently*.

        Must ignore EOS so every config produces the same number of tokens, must not
        reuse cached prefixes, and must return only after the GPU work is complete.
        """

    def reset_peak(self) -> None:
        """Reset any stack-internal peak-memory counters."""

    def extra_metrics(self) -> dict:
        return {}

    # ---- quality hooks -----------------------------------------------------------
    def score_windows(self, windows: list[list[int]]) -> list[tuple[float, int]]:
        """Per window: (sum of -log p over tokens 1..L-1, number of scored tokens)."""
        raise NotImplementedError(f"{type(self).__name__} has no perplexity scorer")

    def next_token_topk_batch(self, prompts: list[list[int]], k: int = 20) -> list[list[tuple[str, float]]]:
        """For each prompt, the top-k (decoded token text, logprob) of the next token."""
        raise NotImplementedError(f"{type(self).__name__} has no next-token scorer")

    def close(self) -> None:
        """Release the GPU."""
