"""Loads the experiment settings from configs/matrix.yaml.

This file only uses light imports so every virtualenv (HF, vLLM, llama.cpp) can use it.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "matrix.yaml"
RESULTS_DIR = ROOT / "results"

STACKS = ("hf", "vllm", "llamacpp")


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: int    # prompt length in tokens
    output_len: int   # tokens to generate


@dataclass
class ModelConfig:
    """One row of the experiment: a model checkpoint run on one stack at one precision."""
    id: str
    stack: str
    precision: str
    checkpoint: str
    tokenizer: str
    quantization: str | None = None
    file: str | None = None          # GGUF file name inside the checkpoint repo (llama.cpp only)
    enabled: bool = True
    notes: str = ""


@dataclass
class Matrix:
    model_name: str
    tokenizer: str
    workloads: dict[str, Workload]
    concurrency: list[int]
    settings: dict
    configs: list[ModelConfig] = field(default_factory=list)

    def get(self, config_id: str) -> ModelConfig:
        for c in self.configs:
            if c.id == config_id:
                return c
        raise KeyError(f"unknown config {config_id!r}; known: {[c.id for c in self.configs]}")

    def enabled(self) -> list[ModelConfig]:
        return [c for c in self.configs if c.enabled]


def load_matrix(path: str | pathlib.Path = DEFAULT_MATRIX) -> Matrix:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    tok = raw["model"]["tokenizer"]
    workloads = {k: Workload(k, v["input_len"], v["output_len"]) for k, v in raw["workloads"].items()}
    configs = []
    for c in raw["configs"]:
        if c["stack"] not in STACKS:
            raise ValueError(f"config {c['id']}: unknown stack {c['stack']!r}")
        configs.append(ModelConfig(tokenizer=tok, **c))
    ids = [c.id for c in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate config ids in matrix")
    return Matrix(
        model_name=raw["model"]["name"],
        tokenizer=tok,
        workloads=workloads,
        concurrency=list(raw["concurrency"]),
        settings=dict(raw.get("settings", {})),
        configs=configs,
    )
