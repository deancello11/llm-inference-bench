from __future__ import annotations

from ..config import ModelConfig
from .base import Adapter, GenResult


def make_adapter(cfg: ModelConfig, settings: dict) -> Adapter:
    """Lazy imports: each stack's venv only needs its own heavy dependencies."""
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
