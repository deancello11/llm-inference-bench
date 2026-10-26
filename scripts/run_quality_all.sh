#!/usr/bin/env bash
# Runs the full quality tests (perplexity, ARC, MMLU, HellaSwag, plus a repeatability check)
# for every config, each with the Python of its own stack.
#
# Run this AFTER the speed benchmark. These tests use the GPU, so running them at the same
# time would change the speed and energy numbers. If one config fails, it is reported and
# the script moves on to the next.
#
#   export BENCH_PY_HF=... BENCH_PY_VLLM=... BENCH_PY_LLAMACPP=...   (see setup_envs.sh)
#   bash scripts/run_quality_all.sh
set -u
cd "$(dirname "$0")/.."

for c in $("$BENCH_PY_HF" -c "from bench.config import load_matrix as l;print(*[c.id for c in l().enabled()])"); do
  # The config id starts with its stack name (hf-..., vllm-..., llamacpp-...)
  case "${c%%-*}" in
    hf) py=$BENCH_PY_HF ;;
    vllm) py=$BENCH_PY_VLLM ;;
    llamacpp) py=$BENCH_PY_LLAMACPP ;;
    *) echo "!! unknown stack for $c"; continue ;;
  esac
  echo "=== quality: $c"
  PATH="$(dirname "$py"):$PATH" "$py" -m bench.quality.run_quality --config "$c" --determinism-check \
    || echo "!! quality failed for $c"
done
