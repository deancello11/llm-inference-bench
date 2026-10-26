"""Hugging Face Transformers adapter (FP16, bitsandbytes INT8, bitsandbytes NF4).

For concurrency N, we send one fixed batch of N prompts of the same length. Transformers
can't add new requests to a running batch, which is part of what this benchmark shows.
"""
from __future__ import annotations

import time

from .base import Adapter, GenResult


class HFAdapter(Adapter):
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        c = self.cfg
        qcfg = None
        if c.quantization == "int8":
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        elif c.quantization == "nf4":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif c.quantization is not None:
            raise ValueError(f"hf adapter: unknown quantization {c.quantization!r}")

        kw = dict(device_map={"": 0}, quantization_config=qcfg, attn_implementation="sdpa")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(c.checkpoint, dtype=torch.float16, **kw)
        except TypeError:  # older transformers versions call this argument torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(c.checkpoint, torch_dtype=torch.float16, **kw)
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(c.tokenizer)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.load_info = {"weights_bytes": int(self.model.get_memory_footprint())}

    # ---- speed test --------------------------------------------------------------
    def generate(self, prompts, max_new_tokens, keep_text=False):
        torch = self.torch
        from transformers import LogitsProcessor, LogitsProcessorList

        stamps: list[float] = []

        class StepTimer(LogitsProcessor):
            """Records the time after every generation step.

            The first stamp is the end of prompt processing, which gives the time to first
            token. We wait for the GPU to finish first so the time is real.
            """

            def __call__(self, input_ids, scores):
                torch.cuda.synchronize()
                stamps.append(time.perf_counter())
                return scores

        ids = torch.tensor(prompts, dtype=torch.long, device="cuda")
        attn = torch.ones_like(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,   # blocks the end-of-text token so the length is exact
                do_sample=False,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                pad_token_id=self.pad_id,
                logits_processor=LogitsProcessorList([StepTimer()]),
            )
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        n_gen = out.shape[1] - ids.shape[1]
        ttft = stamps[0] - t0
        e2e = t_end - t0
        texts = self.tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True) if keep_text else [None] * len(prompts)
        return [GenResult(ttft, e2e, n_gen, t) for t in texts]

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats()

    def extra_metrics(self):
        t = self.torch.cuda
        return {
            "torch_peak_allocated_mib": t.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": t.max_memory_reserved() / 2**20,
        }

    # ---- quality tests -----------------------------------------------------------
    def score_windows(self, windows):
        torch = self.torch
        import torch.nn.functional as F

        res = []
        with torch.inference_mode():
            for w in windows:
                ids = torch.tensor([w], device="cuda")
                logits = self.model(input_ids=ids).logits[0, :-1].float()
                nll = F.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
                res.append((nll, len(w) - 1))
        return res

    def next_token_topk_batch(self, prompts, k=20):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for p in prompts:
                ids = torch.tensor([p], device="cuda")
                lp = torch.log_softmax(self.model(input_ids=ids).logits[0, -1].float(), dim=-1)
                top = torch.topk(lp, k)
                out.append([(self.tok.decode([i]), v) for i, v in zip(top.indices.tolist(), top.values.tolist())])
        return out

    def close(self):
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
