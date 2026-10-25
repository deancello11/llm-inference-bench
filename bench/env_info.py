"""Record the software/hardware environment alongside results (results depend on it)."""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys

from .config import ROOT

PACKAGES = [
    "torch", "transformers", "bitsandbytes", "vllm", "accelerate", "autoawq",
    "auto-gptq", "gptqmodel", "llama-cpp-python", "nvidia-ml-py", "lm_eval",
    "zeus", "zeus-ml", "datasets", "numpy", "httpx",
]


def _run(cmd: list[str]) -> str | None:
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
        import torch  # noqa: WPS433 - optional
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        pass
    return env
