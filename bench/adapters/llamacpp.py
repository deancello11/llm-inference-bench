"""llama.cpp adapter. It runs `llama-server` and sends it requests over HTTP.

The server is restarted whenever the concurrency level changes, with one slot for each
simultaneous request. Prompts are sent as token ids so the length is exact, prompt caching is
turned off, and the end-of-text token is ignored so every reply has the same length.

Safety check: llama.cpp quietly falls back to the CPU if the GPU isn't used, and then
looks about 10x slower. So we refuse to benchmark unless GPU memory grew by roughly the
size of the model after the server started.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..config import RESULTS_DIR, ROOT
from .base import Adapter, GenResult


def find_binary(name: str, env_var: str) -> str:
    if os.environ.get(env_var):
        return os.environ[env_var]
    exe = name + (".exe" if os.name == "nt" else "")
    local = ROOT / "llama.cpp" / "build" / "bin" / exe
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found. Build llama.cpp (scripts/setup_envs.sh) or set ${env_var}.")


def _gpu_used_mib() -> float:
    """GPU memory currently in use, in MiB."""
    import pynvml

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 2**20


_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def gguf_shards(cfg) -> list[str]:
    """Download the GGUF file(s) for this config and return their local paths.

    Big models can be split into parts named like ...-00001-of-00002.gguf. We fetch all parts.
    """
    from huggingface_hub import hf_hub_download

    names = [cfg.file]
    m = _SHARD.search(cfg.file)
    if m:
        total = int(m.group(2))
        names = [f"{cfg.file[:m.start()]}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    return [hf_hub_download(cfg.checkpoint, n) for n in names]


def gguf_path(cfg) -> str:
    """The path to give llama.cpp. For split models that is the first part, and it finds the rest."""
    return gguf_shards(cfg)[0]


class LlamaCppAdapter(Adapter):
    PORT = 8089

    def load(self) -> None:
        self.server_bin = find_binary("llama-server", "LLAMA_SERVER_BIN")
        shards = gguf_shards(self.cfg)
        self.model_path = shards[0]
        self.proc: subprocess.Popen | None = None
        self.parallel = 0
        self.log_dir = pathlib.Path(RESULTS_DIR) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.load_info = {
            "weights_bytes": sum(os.path.getsize(p) for p in shards),
            "gguf_file": self.cfg.file,
            "server_bin": self.server_bin,
        }

    # ---- starting and stopping the server ----------------------------------------
    def _stop_server(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            time.sleep(2)  # give the GPU a moment to free its memory

    def _start_server(self, parallel: int):
        self._stop_server()
        mem_before = _gpu_used_mib()
        s = self.settings
        log_path = self.log_dir / f"llama-server_{self.cfg.id}_np{parallel}.log"
        cmd = [
            self.server_bin, "-m", self.model_path,
            "-ngl", str(s["llamacpp_ngl"]),
            "--parallel", str(parallel),
            "--ctx-size", str(parallel * s["llamacpp_slot_ctx"]),
            "--host", "127.0.0.1", "--port", str(self.PORT),
        ]
        self._log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + 300   # wait up to 5 minutes for the server to be ready
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {log_path}")
            try:
                if httpx.get(f"http://127.0.0.1:{self.PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"llama-server did not become healthy; see {log_path}")
        self.parallel = parallel
        self._check_gpu_offload(log_path, mem_before)

    def _check_gpu_offload(self, log_path: pathlib.Path, mem_before_mib: float):
        """Stop the run if the model did not really load onto the GPU.

        Main check: GPU memory must have grown by at least 80% of the model size. This works
        no matter how llama.cpp words its log lines, which change between versions.
        If this build prints the old "offloaded N/N layers" line, we check that too.
        """
        text = log_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            self.load_info["gpu_layers"] = f"{done}/{total}"
            if done != total:
                raise RuntimeError(f"llama.cpp offloaded only {done}/{total} layers to GPU; refusing to benchmark")
        delta = _gpu_used_mib() - mem_before_mib
        need = 0.8 * self.load_info["weights_bytes"] / 2**20
        self.load_info["gpu_mem_delta_mib"] = round(delta)
        if delta < need:
            raise RuntimeError(f"GPU memory grew only {delta:.0f} MiB after loading a {need / 0.8:.0f} MiB model "
                               f"(need >= {need:.0f}); llama.cpp is not using the GPU. See {log_path}")

    def prepare(self, concurrency: int):
        if self.parallel != concurrency:
            self._start_server(concurrency)

    # ---- speed test --------------------------------------------------------------
    def _one(self, prompt: list[int], n: int, t0: float, keep_text: bool) -> GenResult:
        body = {
            "prompt": prompt, "n_predict": n, "stream": True,
            "temperature": 0.0, "top_k": 1, "repeat_penalty": 1.0,
            "ignore_eos": True, "cache_prompt": False,
        }
        first = None
        n_tok = 0
        text = []
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", f"http://127.0.0.1:{self.PORT}/completion", json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    if first is None:
                        first = time.perf_counter()
                    if ev.get("stop"):
                        n_tok = ev.get("tokens_predicted", n_tok)
                    else:
                        text.append(ev.get("content", ""))
        end = time.perf_counter()
        return GenResult(first - t0, end - t0, n_tok, "".join(text) if keep_text else None)

    def generate(self, prompts, max_new_tokens, keep_text=False):
        assert self.parallel >= len(prompts), "call prepare(concurrency) first"
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            futs = [ex.submit(self._one, p, max_new_tokens, t0, keep_text) for p in prompts]
            return [f.result() for f in futs]

    # ---- quality tests -----------------------------------------------------------
    def next_token_topk_batch(self, prompts, k=20):
        out = []
        with httpx.Client(timeout=120) as client:
            for p in prompts:
                body = {"prompt": p, "n_predict": 1, "temperature": 0.0, "n_probs": k,
                        "cache_prompt": False, "post_sampling_probs": False}
                r = client.post(f"http://127.0.0.1:{self.PORT}/completion", json=body)
                r.raise_for_status()
                cp = r.json()["completion_probabilities"][0]
                if "top_logprobs" in cp:                       # newer llama.cpp servers
                    out.append([(e["token"], e["logprob"]) for e in cp["top_logprobs"]])
                else:                                          # older llama.cpp servers
                    import math
                    out.append([(e["tok_str"], math.log(max(e["prob"], 1e-30))) for e in cp["probs"]])
        return out

    def close(self):
        self._stop_server()
