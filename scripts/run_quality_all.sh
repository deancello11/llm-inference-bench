#!/usr/bin/env bash
# Full quality evals (ppl, arc, mmlu, hellaswag + a determinism check) for every enabled config,
# each with its own stack's Python. Run AFTER the benchmark matrix: it uses the GPU, so running
# it alongside would distort the latency/energy numbers. A failing config is logged, not fatal.
#
#   export BENCH_PY_HF=... BENCH_PY_VLLM=... BENCH_PY_LLAMACPP=...   (see setup_envs.sh)
#   bash scripts/run_quality_all.sh
set -u
cd "$(dirname "$0")/.."

for c in $("$BENCH_PY_HF" -c "from bench.config import load_matrix as l;print(*[c.id for c in l().enabled()])"); do
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
