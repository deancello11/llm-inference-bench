# Quantization benchmark: speed, energy and quality of a 3B model on one GPU

I recently took a machine learning course and an energy efficient computing course in college, and I wanted
to build something that used what I learned in both. This project is the result. I wanted to know what you
actually get when you shrink a language model. How much faster does it run, how much less energy does it use,
and how much worse do its answers get? To find out, I ran Qwen2.5-3B-Instruct on a single RTX 4090 in 9 different
setups and measured all three.

The short version: 4-bit models generate text about 3x faster than the 16-bit Hugging Face baseline and use about
two-thirds less energy per token, and they lose between 1 and 3 points of accuracy. The software you run the model with matters just
as much as the precision, though. One result surprised me: the 8-bit mode in bitsandbytes is much slower than plain 16-bit.

The full explanation is in [docs/WRITEUP.md](docs/WRITEUP.md).

![Accuracy vs energy per token](results/charts/pareto.png)

## Background

**Quantization.** A model is a huge list of numbers called weights. Normally each one is stored in 16 bits.
Quantization stores them in fewer bits, like 8 or 4. It is like saving a photo at lower quality: the file gets much
smaller and usually looks almost the same. A smaller model also tends to run faster, because the GPU has less
data to read for every word it writes. The cost is a small loss in accuracy.

**The three stacks.** These are the programs that load a model and run it on the GPU.
- **Hugging Face Transformers (HF):** the most common one and easy to use. I used it as the baseline.
- **vLLM:** built for speed when many requests arrive at once, like a real chatbot service.
- **llama.cpp:** a lightweight program that runs models almost anywhere. It uses its own file format, GGUF.

**The 4-bit methods.** All of these squeeze a model to about 4 bits per weight, but they choose the numbers differently.
- **NF4 (bitsandbytes):** simple rounding with cleverly spaced steps. No calibration data.
- **AWQ:** runs sample text through the model to find the weights that matter most, and protects them before rounding.
- **GPTQ:** rounds one weight at a time and adjusts the remaining weights to cancel out the error.
- **Q4_K_M (llama.cpp):** llama.cpp's own format, a mix of 4-bit and 6-bit blocks.

Config names look like `stack-precision`, for example `vllm-awq`. `fp16` is the full 16-bit model, `int8` and `q8` are 8-bit.

## Results

Each row is one config. "c=1" means one request at a time (like a single user), and "c=32" means 32 requests at once
(like a busy service). The prompt is 128 tokens and each reply is exactly 256 tokens. Every number is the median
of 6 timed runs.

| config | tokens/s (c=1) | mJ per token (c=1) | tokens/s (c=32) | mJ per token (c=32) | model size GB | accuracy % | perplexity |
|---|---|---|---|---|---|---|---|
| hf-fp16 | 88 | 2640 | 2449 | 113 | 6.17 | 75.3 | 8.56 |
| hf-int8-bnb | 19 | 4700 | 482 | 198 | 3.40 | 74.8 | 8.66 |
| hf-nf4-bnb | 61 | 1965 | 1475 | 147 | 2.01 | 71.3 | 9.10 |
| vllm-fp16 | 132 | 2282 | 3351 | 100 | 6.17 | 75.2 | 8.56 |
| vllm-awq | 280 | 906 | 6317 | 54 | 2.69 | 72.5 | 9.10 |
| vllm-gptq | 284 | 885 | 6352 | 53 | 2.07 | 72.5 | 9.22 |
| llamacpp-f16 | 133 | 2197 | 1437 | 126 | 6.80 | 74.9 | 7.73* |
| llamacpp-q8 | 215 | 1330 | 1625 | 96 | 3.62 | 75.0 | 7.75* |
| llamacpp-q4km | 309 | 989 | 1758 | 85 | 2.10 | 74.0 | 7.98* |

- **mJ per token** is millijoules of GPU energy for each generated token. Lower is better.
- **accuracy** is the average of three tests: ARC-Challenge, MMLU and HellaSwag. A difference under about 1 point is just noise.
- **perplexity** is how well the model predicts normal text (WikiText-2). Lower is better.
- \*llama.cpp scores perplexity its own way, so only compare those three numbers with each other.

What stands out:
- The best 4-bit setups (vLLM AWQ and GPTQ, llama.cpp Q4_K_M) are about 2.2x faster than their own 16-bit versions, and cost about 1 to 3 accuracy points.
- With many requests at once, vLLM is far ahead: about 6,300 tokens/s with AWQ or GPTQ, against 1,400 to 1,800 for llama.cpp.
- Batching saves a lot of energy. Going from 1 request to 32 cut the energy per token by 12 to 24 times.
- bitsandbytes was the slowest. 8-bit ran at 19 tokens/s against 88 for 16-bit, even though it reads half as many bytes.

The long-prompt results (1024 tokens), the concurrency 8 results and every individual run are in
`results/runs.jsonl` and `results/charts/summary.csv`. More charts are in `results/charts/`.

## How I measured

- **Same input everywhere.** Every stack gets the same kind of random token ids, so the prompt length is exact. Each request
  is different, and prompt caching is turned off, so nothing gets reused.
- **Same output length.** Greedy decoding, and the end-of-text token is blocked, so every reply is exactly 256 tokens.
- **Energy.** I read the GPU's built-in energy counter before and after each run and divided by the tokens generated.
  This only counts the GPU, not the CPU, memory or the wall socket. I also logged power readings during the run, and they
  agree with the counter on average (within 0.1%, with 12 of 324 runs off by more than 5%).
- **Timed runs.** Each run lasts at least 10 seconds so the energy reading is steady. I threw away a warmup run first, did
  6 runs per setting, and shuffled the order of configs each round so a warming GPU didn't favor one stack. No run was
  slowed down by heat.
- **Quality.** Perplexity on WikiText-2, plus ARC-Challenge, MMLU and HellaSwag. All three stacks answer the same multiple-choice
  prompts the same way, so the accuracy is comparable. Scoring 100 questions twice gave identical answers for every config.
- **Memory.** vLLM grabs a fixed share of GPU memory up front, so its total memory use says nothing. I report the size of
  the model and the size of the KV cache instead.

## My setup

| | |
|---|---|
| GPU | NVIDIA RTX 4090 (24 GB), rented on RunPod, driver 595.91.07 |
| Hugging Face | torch 2.14.1, transformers 5.20.2, bitsandbytes 0.50.2 |
| vLLM | vllm 0.32.0, torch 2.13.0 |
| llama.cpp | build 11889 (commit `d854d98`), built with CUDA, all layers on the GPU |

`results/loads.jsonl` stores the full list of versions for every run. The models are the official Qwen2.5-3B-Instruct
checkpoints (plain, AWQ, GPTQ-Int4 and GGUF). The HF 8-bit and NF4 versions are made from the 16-bit model when it loads.

## Run it yourself

You need a Linux machine with an NVIDIA GPU. Everything is run from the repo folder.

```bash
bash scripts/setup_envs.sh              # makes 3 virtualenvs and builds llama.cpp with CUDA
export HF_HOME=/workspace/hf            # where to keep downloaded models
python scripts/check_gpu.py             # checks the GPU gives real power readings. Do this first.
python scripts/download_models.py       # downloads all the models
python -m bench.sanity --config hf-fp16 # prints a few real answers. Check each config this way.
python -m bench.run_matrix              # the full speed benchmark (a few hours, safe to restart)
bash scripts/run_quality_all.sh         # the quality tests, run after the benchmark
python -m analysis.make_charts          # makes the tables and charts (works on a laptop)
```

Set `BENCH_PY_HF`, `BENCH_PY_VLLM` and `BENCH_PY_LLAMACPP` to each virtualenv's Python first. `setup_envs.sh` prints how.
You can test the parts that don't need a GPU with `python -m pytest tests`.

## Limits

- One GPU, one model, and short test prompts made of random tokens. Results may differ on other hardware or models.
- Energy is the GPU only.
- The 4-bit methods are not identical to each other, so the comparison between them is not perfectly fair.
- My accuracy test is a simple multiple-choice setup, so the numbers are not comparable to public leaderboards.
- I did not run the Zeus energy cross-check (`scripts/zeus_crosscheck.py`), and I did not profile why the slow configs
  are slow. The explanations in the writeup come from the measured timings.

## Files

```
configs/matrix.yaml      the experiment: which models and settings to test
bench/                   the benchmark code (one small adapter per stack, GPU monitor, runners)
bench/quality/           perplexity and accuracy tests
scripts/                 setup, GPU check, model download, quality runner
analysis/make_charts.py  turns the results into tables and charts
results/                 raw data from my runs, plus the charts
docs/WRITEUP.md          the full write-up
```
