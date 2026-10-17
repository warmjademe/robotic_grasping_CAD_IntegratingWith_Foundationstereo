#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_HOME="$ROOT/.cuda"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export PATH="$CUDA_HOME/bin:$ROOT/.venv-infer/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for header_dir in "$ROOT"/.venv-infer/lib/python3.11/site-packages/nvidia/*/include "$ROOT"/.venv-infer/lib/python3.11/site-packages/cusparselt/include; do
    if [[ -d "$header_dir" ]]; then
        CPATH="$CPATH:$header_dir"
    fi
done
"$ROOT/ops/uv" pip install --python "$ROOT/.venv-infer/bin/python" --no-build-isolation --no-deps "$ROOT/upstream/GraspGen/pointnet2_ops"
