"""The interface every serving stack (Hugging Face, vLLM, llama.cpp) has to follow.

The benchmark only ever calls generate(), so all stacks are timed by the same code.
The quality tests use score_windows() and next_token_topk_batch().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import ModelConfig


@dataclass
class GenResult:
    ttft_s: float          # time from sending the request to the first generated token
    e2e_s: float           # time from sending the request to the last token
    n_tokens: int          # tokens generated (must equal the requested length)
    text: str | None = None


class Adapter(ABC):
    def __init__(self, cfg: ModelConfig, settings: dict):
        self.cfg = cfg
        self.settings = settings
        self.load_info: dict = {}

    @abstractmethod
    def load(self) -> None:
        """Load the model onto the GPU and fill self.load_info (weight size, and so on)."""

    def prepare(self, concurrency: int) -> None:
        """Called before each concurrency level. llama.cpp restarts its server here."""

    @abstractmethod
    def generate(self, prompts: list[list[int]], max_new_tokens: int, keep_text: bool = False) -> list[GenResult]:
        """Generate exactly max_new_tokens for every prompt, all at the same time.

        Rules every stack must follow: greedy decoding, ignore the end-of-text token so all
        runs have the same length, don't reuse cached prompts, and only return once the
        GPU work is finished.
        """

    def reset_peak(self) -> None:
        """Reset the stack's own peak-memory counter, if it has one."""

    def extra_metrics(self) -> dict:
        return {}

    # ---- used by the quality tests -----------------------------------------------
    def score_windows(self, windows: list[list[int]]) -> list[tuple[float, int]]:
        """For each text window return (total negative log-likelihood, number of tokens scored)."""
        raise NotImplementedError(f"{type(self).__name__} has no perplexity scorer")

    def next_token_topk_batch(self, prompts: list[list[int]], k: int = 20) -> list[list[tuple[str, float]]]:
        """For each prompt return the k most likely next tokens as (text, log-probability)."""
        raise NotImplementedError(f"{type(self).__name__} has no next-token scorer")

    def close(self) -> None:
        """Free the GPU."""
