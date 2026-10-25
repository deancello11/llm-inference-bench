#!/usr/bin/env bash
# OPTIONAL cross-check of our built-in accuracy numbers using EleutherAI lm-evaluation-harness
# (HF and vLLM backends only; its llama.cpp loglikelihood support is unreliable, which is why
# the main results use bench.quality.mc_eval for all three stacks).
#
# Run inside the HF venv for hf-*, inside the vLLM venv for vllm-*:  pip install lm_eval
set -euo pipefail
OUT=results/lm_eval
mkdir -p "$OUT"
TASKS=arc_challenge,hellaswag,mmlu
LIMIT=${LIMIT:-}   # e.g. LIMIT=0.1 for a 10% subset

lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16 \
  --tasks $TASKS ${LIMIT:+--limit $LIMIT} --output_path "$OUT/hf-fp16"
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16,load_in_8bit=True \
  --tasks $TASKS ${LIMIT:+--limit $LIMIT} --output_path "$OUT/hf-int8-bnb"
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16,load_in_4bit=True,bnb_4bit_quant_type=nf4 \
  --tasks $TASKS ${LIMIT:+--limit $LIMIT} --output_path "$OUT/hf-nf4-bnb"
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-3B-Instruct-AWQ,quantization=awq,dtype=float16,gpu_memory_utilization=0.6,max_model_len=4096 \
  --tasks $TASKS ${LIMIT:+--limit $LIMIT} --output_path "$OUT/vllm-awq"
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4,quantization=gptq,dtype=float16,gpu_memory_utilization=0.6,max_model_len=4096 \
  --tasks $TASKS ${LIMIT:+--limit $LIMIT} --output_path "$OUT/vllm-gptq"
