"""vLLM adapter (FP16, AWQ, GPTQ) using the offline engine with manual stepping.

We drive `llm_engine.add_request()/step()` ourselves so we can timestamp each request's
first token (TTFT) without depending on version-specific `RequestOutput.metrics`.
Continuous batching + PagedAttention are exactly as in the production server.
"""
from __future__ import annotations

import time
import uuid

from ..io_utils import checkpoint_bytes
from .base import Adapter, GenResult


class VLLMAdapter(Adapter):
    def load(self) -> None:
        from vllm import LLM

        s, c = self.settings, self.cfg
        self.llm = LLM(
            model=c.checkpoint,
            tokenizer=c.tokenizer,
            quantization=c.quantization,
            dtype="float16",
            gpu_memory_utilization=s["vllm_gpu_memory_utilization"],
            max_model_len=s["vllm_max_model_len"],
            max_num_seqs=s["vllm_max_num_seqs"],
            enable_prefix_caching=False,  # otherwise repeated prefixes make prefill look free
            seed=0,
        )
        info = {"weights_bytes": checkpoint_bytes(c.checkpoint),
                "gpu_memory_utilization": s["vllm_gpu_memory_utilization"]}
        try:  # KV-cache capacity (version-dependent attribute paths)
            eng = self.llm.llm_engine
            cc = getattr(eng, "cache_config", None) or eng.vllm_config.cache_config
            info["kv_cache_blocks"] = cc.num_gpu_blocks
            info["kv_cache_block_size"] = cc.block_size
            info["kv_cache_tokens"] = cc.num_gpu_blocks * cc.block_size
        except Exception as e:  # noqa: BLE001
            info["kv_cache_error"] = repr(e)
        self.load_info = info

    def generate(self, prompts, max_new_tokens, keep_text=False):
        from vllm import SamplingParams

        eng = self.llm.llm_engine
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
        tag = uuid.uuid4().hex[:8]
        rids = [f"{tag}-{i}" for i in range(len(prompts))]
        first: dict[str, float] = {}
        done: dict[str, tuple[float, int, str]] = {}
        t0 = time.perf_counter()
        for rid, p in zip(rids, prompts):
            eng.add_request(rid, {"prompt_token_ids": p}, params)
        while eng.has_unfinished_requests():
            outs = eng.step()
            now = time.perf_counter()
            for o in outs:
                n = len(o.outputs[0].token_ids)
                if n > 0 and o.request_id not in first:
                    first[o.request_id] = now
                if o.finished:
                    done[o.request_id] = (now, n, o.outputs[0].text)
        return [
            GenResult(first[r] - t0, done[r][0] - t0, done[r][1], done[r][2] if keep_text else None)
            for r in rids
        ]

    # ---- quality path ------------------------------------------------------------
    def score_windows(self, windows):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate([{"prompt_token_ids": w} for w in windows], params, use_tqdm=False)
        res = []
        for w, o in zip(windows, outs):
            nll, n = 0.0, 0
            for tok_id, lp in zip(w[1:], o.prompt_logprobs[1:]):
                nll -= lp[tok_id].logprob
                n += 1
            res.append((nll, n))
        return res

    def next_token_topk_batch(self, prompts, k=20):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=k)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        return [[(lp.decoded_token, lp.logprob) for lp in o.outputs[0].logprobs[0].values()] for o in outs]

    def close(self):
        import gc

        try:
            del self.llm
            gc.collect()
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
