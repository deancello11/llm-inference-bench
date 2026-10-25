#!/usr/bin/env bash
# Create one virtualenv per serving stack (vLLM pins its own torch, so they must be separate)
# and build llama.cpp with CUDA. Run on the rented Linux GPU pod from the repo root.
#
#   bash scripts/setup_envs.sh
#   export BENCH_PY_HF=$PWD/.venv-hf/bin/python BENCH_PY_VLLM=$PWD/.venv-vllm/bin/python \
#          BENCH_PY_LLAMACPP=$PWD/.venv-llamacpp/bin/python
set -euo pipefail

for stack in hf vllm llamacpp; do
  python3 -m venv ".venv-$stack"
  ".venv-$stack/bin/pip" install -U pip
  ".venv-$stack/bin/pip" install -r requirements/common.txt -r "requirements/$stack.txt"
  ".venv-$stack/bin/pip" freeze > "requirements/lock-$stack.txt"   # commit these: pinned versions you actually ran
done

if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp; fi
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j"$(nproc)" --target llama-server llama-perplexity
git -C llama.cpp rev-parse HEAD > requirements/llamacpp-commit.txt
echo "done. Next: python scripts/check_gpu.py"
