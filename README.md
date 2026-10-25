# LLM inference efficiency & energy benchmark

**Question:** for Qwen2.5-3B-Instruct on one RTX 4090, how do quantization level and serving stack
(Hugging Face, vLLM, llama.cpp) trade off latency, throughput, memory, **energy per token**, and answer quality?

> Results: _not run yet._ After a run, paste `results/charts/pareto.png` and
> `results/charts/summary_short_c1.md` here, and fill in `docs/WRITEUP.md` from `docs/WRITEUP_TEMPLATE.md`.

## Configs (9)

| id | stack | weights | checkpoint |
|---|---|---|---|
| hf-fp16 / hf-int8-bnb / hf-nf4-bnb | Transformers | FP16 / bnb INT8 / bnb NF4 | Qwen/Qwen2.5-3B-Instruct |
| vllm-fp16 / vllm-awq / vllm-gptq | vLLM | FP16 / AWQ int4 / GPTQ int4 | official Qwen checkpoints |
| llamacpp-f16 / -q8 / -q4km | llama-server | GGUF F16 / Q8_0 / Q4_K_M | Qwen/Qwen2.5-3B-Instruct-GGUF |

Edit `configs/matrix.yaml` to add/remove anything. Workloads: 128-in/256-out and 1024-in/256-out, concurrency 1/8/32.

## Layout

```
configs/matrix.yaml        the experiment definition
bench/adapters/            one thin adapter per stack (same generate() interface)
bench/gpu.py               NVML energy counter + power/clock/temp/memory sampler
bench/run_config.py        measure one config (warmup, repeats, raw JSONL rows)
bench/run_matrix.py        whole grid: resumable, interleaved rounds, per-stack venvs
bench/quality/             WikiText-2 perplexity + ARC/MMLU/HellaSwag accuracy
bench/sanity.py            read real generations before trusting numbers
scripts/                   setup, GPU gate check, model download, Zeus cross-check, lm-eval
analysis/make_charts.py    summary table + all charts (runs on a laptop)
results/                   raw data: runs.jsonl, loads.jsonl, quality.jsonl, failures.jsonl
```

## Run order (on the rented 4090 pod, Linux)

```bash
bash scripts/setup_envs.sh                     # 3 venvs + CUDA build of llama.cpp; then export BENCH_PY_* (printed in the script header)
export HF_HOME=/workspace/hf                   # persistent storage
python scripts/check_gpu.py                    # GATE: full GPU, real power, energy counter agrees with integrated power
python scripts/download_models.py
python -m bench.sanity --config hf-fp16        # repeat per config; read the text
python -m bench.run_config --config hf-fp16 --workloads short --concurrency 1 --repeats 3   # first end-to-end check
python scripts/zeus_crosscheck.py              # energy validation (HF env, pip install zeus)
python -m bench.quality.run_quality --config hf-fp16 --tasks arc --limit 50                 # smoke test, then drop --limit
python -m bench.run_matrix --rounds 3 --repeats-per-load 2                                  # the full grid, resumable
# quality for every config (one call each):
for c in $(python -c "from bench.config import load_matrix as l;print(*[c.id for c in l().enabled()])"); do
  python -m bench.quality.run_quality --config $c; done
# on your laptop:
pip install -r requirements/analysis.txt && python -m analysis.make_charts   # add --energy idle_sub for idle-subtracted
```

Local, no-GPU check of everything that can be checked: `python -m pytest tests` (includes a synthetic-data chart render).

## Methodology

- **Same tokens everywhere.** Prompts are seeded random token ids sent to every stack as ids, so input length is exact;
  they differ per request and prefix caching is disabled (vLLM `enable_prefix_caching=False`, llama.cpp `cache_prompt=false`)
  so prefill is never a cache hit.
- **Exact output length.** Greedy decoding, EOS ignored/suppressed, 256 tokens; a run with any other length is rejected.
- **Concurrency.** vLLM and llama.cpp: N simultaneous requests under continuous batching. Hugging Face: one static batch of N
  (it has no continuous batching, which is part of the comparison).
- **TTFT / decode split.** HF: a logits-processor timestamps each step (first call = end of prefill). vLLM: timestamps around
  `engine.step()`. llama.cpp: first SSE event. Decode tok/s = (n-1)/(e2e - TTFT) per request.
- **Measurement windows.** Each repeat loops generate() until >= 10 s so the energy reading is stable; latency is pooled over
  every request in the window. Warmup is discarded. Every window is one JSONL row.
- **Energy.** NVML cumulative energy counter read right before/after the window, divided by generated tokens (prefill included).
  Cross-checked against integrated 10 Hz power samples (in every row) and Zeus. Idle power is measured after load and stored;
  charts default to **raw** energy, `--energy idle_sub` gives idle-subtracted. **Scope: GPU board only** (no CPU/DRAM/wall).
- **Throttling.** SM clock, temperature and throttle-reason bits are logged per window; thermal/HW slowdowns are flagged and
  counted in the summary. Repeats are spread over rounds with shuffled config order so drift doesn't bias one stack.
- **Memory.** Reported as (a) weight bytes (checkpoint size / `get_memory_footprint` / GGUF size) and (b) peak device memory
  delta from NVML. vLLM preallocates `gpu_memory_utilization=0.60` of VRAM, so its (b) is a setting, not a measurement; its
  KV-cache capacity is logged in `loads.jsonl`.
- **Quality.** Perplexity: WikiText-2 test, non-overlapping 2048-token windows (HF and vLLM identical protocol). llama.cpp uses
  its own `llama-perplexity` (different scoring), so compare it only within llama.cpp. Accuracy: ARC-Challenge (full test),
  MMLU (20/subject, seeded), HellaSwag (1000, seeded) with a single zero-shot letter-choice protocol scored identically on all
  three stacks. These are not leaderboard numbers; compare *deltas between configs*. `scripts/run_lm_eval.sh` gives an optional lm-eval cross-check on HF/vLLM.

## Limitations (state these in any writeup)

One GPU model, one base model, GPU-only energy, quantization schemes that are not equivalent (AWQ/GPTQ int4 vs NF4 vs Q4_K_M
at ~4.9 bits/weight), synthetic random-token prompts, and quality numbers from a custom letter-choice protocol.
Findings apply to this model, this GPU, these workloads.
