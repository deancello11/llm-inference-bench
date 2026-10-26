#!/usr/bin/env bash
# Sets up everything on a fresh Linux GPU machine. Run it from the repo folder:
#
#   bash scripts/setup_envs.sh
#
# It makes one Python virtualenv for each stack (vLLM needs its own version of torch, so
# they can't share one) and builds llama.cpp with CUDA support. When it finishes, run:
#
#   export BENCH_PY_HF=$PWD/.venv-hf/bin/python BENCH_PY_VLLM=$PWD/.venv-vllm/bin/python \
#          BENCH_PY_LLAMACPP=$PWD/.venv-llamacpp/bin/python
set -euo pipefail

for stack in hf vllm llamacpp; do
  python3 -m venv ".venv-$stack"
  ".venv-$stack/bin/pip" install -U pip
  ".venv-$stack/bin/pip" install -r requirements/common.txt -r "requirements/$stack.txt"
  # Save the exact versions that got installed, so the setup can be repeated later.
  ".venv-$stack/bin/pip" freeze > "requirements/lock-$stack.txt"
done

if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp; fi
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j"$(nproc)" --target llama-server llama-perplexity
git -C llama.cpp rev-parse HEAD > requirements/llamacpp-commit.txt
echo "Done. Next: python scripts/check_gpu.py"
