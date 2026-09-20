#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT_ROOT=${1:-"$REPO_ROOT/benchmarks/artifacts"}
MYTRITON_PYTHON=${MYTRITON_PYTHON:-python}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$OUTPUT_ROOT/a100-sm80-$RUN_STAMP"
CUPY_CACHE="$RUN_DIR/cupy-cache"

mkdir -p "$CUPY_CACHE"

export CUPY_CACHE_DIR="$CUPY_CACHE"
export CUPY_CACHE_SAVE_CUDA_SOURCE=1

"$MYTRITON_PYTHON" "$SCRIPT_DIR/benchmark_matmul.py" \
    --sizes 4096 \
    --tiles 32x16x32 \
    --dtype bf16 \
    --warmup 3 \
    --repeats 10 \
    --artifacts-dir "$RUN_DIR" \
    --output "$RUN_DIR/benchmark.json" \
    | tee "$RUN_DIR/benchmark.txt"

SOURCE="$RUN_DIR/matmul_bf16_m4096_n4096_k4096_bm32_bk16_bn32.cu"
SOURCE_METADATA="$RUN_DIR/matmul_bf16_m4096_n4096_k4096_bm32_bk16_bn32.json"
NVRTC_CUBIN="$RUN_DIR/matmul_bf16_sm80_nvrtc.cubin"
NVCC_PTX="$RUN_DIR/matmul_bf16_sm80_nvcc.ptx"

mapfile -t CUPY_CACHE_FILES < <(find "$CUPY_CACHE" -maxdepth 1 -type f -name '*.cubin')
if [[ ${#CUPY_CACHE_FILES[@]} -ne 1 ]]; then
    echo "expected one CuPy cache artifact, found ${#CUPY_CACHE_FILES[@]}" >&2
    exit 1
fi

# CuPy prefixes its disk entry with a 40-byte SHA-1 and NVRTC omits the final
# ELF padding byte returned by the driver API. Strip the cache header and add
# the padding byte so command-line CUDA tools can read the exact loaded cubin.
"$MYTRITON_PYTHON" - "${CUPY_CACHE_FILES[0]}" "$NVRTC_CUBIN" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
cached = source.read_bytes()
payload = cached[40:]
if not payload.startswith(b"\x7fELF"):
    raise RuntimeError(f"unexpected CuPy cache format: {source}")
destination.write_bytes(payload + b"\0")
PY

nvcc \
    --std=c++14 \
    --ftz=true \
    --gpu-architecture=compute_80 \
    --ptx "$SOURCE" \
    --output-file "$NVCC_PTX"

cuobjdump --dump-resource-usage "$NVRTC_CUBIN" \
    > "$RUN_DIR/matmul_bf16_sm80_resources.txt"
cuobjdump --dump-sass "$NVRTC_CUBIN" \
    > "$RUN_DIR/matmul_bf16_sm80.sass"

if command -v nvdisasm >/dev/null 2>&1; then
    nvdisasm "$NVRTC_CUBIN" > "$RUN_DIR/matmul_bf16_sm80.nvdisasm"
fi

NCU_REPORT_BASE="$RUN_DIR/matmul_bf16_sm80"
ncu \
    --target-processes all \
    --kernel-name 'regex:.*matmul_kernel.*' \
    --launch-skip 2 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section Occupancy \
    --section LaunchStats \
    --section WarpStateStats \
    --section SchedulerStats \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section SourceCounters \
    --force-overwrite \
    --export "$NCU_REPORT_BASE" \
    "$MYTRITON_PYTHON" "$SCRIPT_DIR/benchmark_matmul.py" \
        --sizes 4096 \
        --tiles 32x16x32 \
        --dtype bf16 \
        --warmup 0 \
        --repeats 1 \
        --no-graphs \
        --output "$RUN_DIR/ncu-benchmark.json" \
    | tee "$RUN_DIR/ncu-capture.txt"

ncu \
    --import "$NCU_REPORT_BASE.ncu-rep" \
    --page details \
    --print-details all \
    > "$RUN_DIR/ncu-details.txt"

"$MYTRITON_PYTHON" - "$SOURCE" "$SOURCE_METADATA" "$NVRTC_CUBIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

source, metadata, cubin = (Path(value) for value in sys.argv[1:])
summary = {
    "generated_cuda": str(source),
    "generated_cuda_sha256": sha256(source.read_bytes()).hexdigest(),
    "source_metadata": str(metadata),
    "nvrtc_cubin": str(cubin),
    "nvrtc_cubin_sha256": sha256(cubin.read_bytes()).hexdigest(),
}
(source.parent / "artifacts.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

echo
echo "A100 profiling artifacts: $RUN_DIR"
