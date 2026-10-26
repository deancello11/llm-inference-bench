"""Saves the software and hardware versions next to the results, since speeds depend on them."""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys

from .config import ROOT

# Packages whose versions we record, if they are installed in the current environment.
PACKAGES = [
    "torch", "transformers", "bitsandbytes", "vllm", "accelerate", "autoawq",
    "auto-gptq", "gptqmodel", "llama-cpp-python", "nvidia-ml-py", "lm_eval",
    "zeus", "zeus-ml", "datasets", "numpy", "httpx",
]


def _run(cmd: list[str]) -> str | None:
    """Run a command and return its output, or None if it fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_env() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,vbios_version,clocks.max.sm,clocks.max.mem,power.limit",
                            "--format=csv,noheader"]),
    }
    try:
        import torch  # only some environments have torch
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return env
