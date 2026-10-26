from __future__ import annotations

from ..config import ModelConfig
from .base import Adapter, GenResult


def make_adapter(cfg: ModelConfig, settings: dict) -> Adapter:
    """Pick the adapter for a config's stack.

    Each import happens inside its branch, so a virtualenv only needs the packages
    for the stack it actually runs.
    """
    if cfg.stack == "hf":
        from .hf import HFAdapter
        return HFAdapter(cfg, settings)
    if cfg.stack == "vllm":
        from .vllm_adapter import VLLMAdapter
        return VLLMAdapter(cfg, settings)
    if cfg.stack == "llamacpp":
        from .llamacpp import LlamaCppAdapter
        return LlamaCppAdapter(cfg, settings)
    raise ValueError(cfg.stack)


__all__ = ["Adapter", "GenResult", "make_adapter"]
