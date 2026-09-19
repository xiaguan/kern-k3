#!/usr/bin/env bash
# Build with CUTLASS 5c149f52 (see PROVENANCE.md).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?out.cubin}"
nvcc="${NVCC:-nvcc}"
arch="${KERN_SM:-sm_103a}"
: "${CUTLASS_INCLUDE:?set CUTLASS_INCLUDE to a CUTLASS include directory}"
"$nvcc" -cubin -O3 -std=c++17 -arch="$arch" \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -I"$here" -I"$here/csrc" -I"$here/csrc/smxx" -I"$CUTLASS_INCLUDE" \
  -o "$out" "$here/kern_flash_kda.cu"
echo "built $out ($arch)" >&2
