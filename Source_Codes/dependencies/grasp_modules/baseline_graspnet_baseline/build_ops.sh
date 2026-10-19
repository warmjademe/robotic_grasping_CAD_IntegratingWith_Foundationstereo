#!/usr/bin/env bash
set -euo pipefail
baseline_dir="$(cd "$(dirname "$0")" && pwd)"
dependency_dir="${GRASPGEN_RUNTIME:?Set GRASPGEN_RUNTIME to the configured runtime}"
export CUDA_HOME="$dependency_dir/.cuda"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for component in cusparse cublas cusolver; do
    export CPATH="$CPATH:$dependency_dir/.venv-infer/lib/python3.11/site-packages/nvidia/$component/include"
done
for component in pointnet2 knn; do
    cd "$baseline_dir/upstream/graspnet-baseline/$component"
    "$baseline_dir/.venv/bin/python" -m pip install --no-deps --no-build-isolation .
done
