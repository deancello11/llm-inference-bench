#!/usr/bin/env bash
# Optional extra check of our accuracy numbers, using EleutherAI's lm-evaluation-harness.
# It only covers Hugging Face and vLLM, because its llama.cpp support isn't reliable. That is
# why the main results use bench/quality/mc_eval.py for all three stacks. This was not run
# for the published results.
#
# Install first:  pip install lm_eval
# Run the hf-* lines in the HF venv and the vllm-* lines in the vLLM venv.
set -euo pipefail
OUT=results/lm_eval
mkdir -p "$OUT"
TASKS=arc_challenge,hellaswag,mmlu
LIMIT=${LIMIT:-}   # for a quick test use a fraction, for example LIMIT=0.1 for 10%

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
