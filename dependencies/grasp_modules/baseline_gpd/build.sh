#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
base="$PWD"
prefix="$base/tooling/mamba-root/envs/env"
export PATH="$prefix/bin:$PATH"
export CMAKE_PREFIX_PATH="$prefix"
export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
cmake -G Ninja -S "$base" -B "$base/build" \
  -DCMAKE_MAKE_PROGRAM="$base/.venv/bin/ninja" \
  -DCMAKE_C_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$base/build" --target gpd_shared_bridge --parallel 4
