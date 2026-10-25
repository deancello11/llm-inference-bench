# Title: What does 4-bit actually buy you? Latency, energy and quality for Qwen2.5-3B on an RTX 4090

(1-2 pages. Fill every [bracket] from results/summary.csv; delete this line.)

## 1. Motivation (3-4 sentences)
Inference is the recurring cost of deploying LLMs. This measures how quantization and serving stack move latency, throughput, energy/token and accuracy.

## 2. Setup in brief
Model, GPU, driver/CUDA, library versions (from `loads.jsonl` env), 9 configs, workloads, energy method (NVML counter, GPU-only), quality protocol. Link to README methodology.

## 3. Headline result
[Pareto chart.] One sentence of the form: "[config] is [X]x faster and uses [Y]% less energy/token than hf-fp16 at a cost of [Z] accuracy points."

## 4. Roofline explanation
- Decode at batch 1 reads every weight once per token: ceiling = bandwidth / weight bytes = 1008 GB/s / [bytes] = [tok/s] per config.
- Measured vs ceiling (roofline.png): [percent of ceiling per config]; explain gaps (kernel launch overhead, dequantization cost, framework overhead, KV reads).
- Prefill is compute-bound: [ttft_vs_input.png]; quantization helps it far less than decode.
- Batching: weight reads are amortized across requests, so energy/token falls [X]x from c=1 to c=32 (energy_vs_concurrency.png) and the gap between configs narrows as decode becomes less bandwidth-bound.

## 5. Surprises and failures
[e.g. bnb INT8 slower than FP16 despite fewer bytes; GPTQ/AWQ kernel selection; any crashes from failures.jsonl.] Each with a mechanism you checked, not a guess.

## 6. Which config would I deploy?
- Latency-sensitive single user: [config], because [TTFT, tok/s at c=1, accuracy cost].
- High-throughput batch service: [config], because [tok/s and mJ/token at c=32, KV capacity].

## 7. Limitations
One GPU, one model, GPU-only energy, non-equivalent quantization schemes, synthetic prompts, custom accuracy protocol.

## 8. Next
Other GPUs, larger models, FP8, speculative decoding, whole-system energy.
