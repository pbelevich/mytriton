#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

REMOTE_ROOT=${MYTRITON_COLAB_ROOT:-/content/mytriton}
COLAB_TIMEOUT=${MYTRITON_COLAB_TIMEOUT:-1800}
COLAB_NOTEBOOK=${COLABRUN_NOTEBOOK:-$REPO_ROOT/tests/mytriton_colab_tests.ipynb}

usage() {
    cat <<'EOF'
Run benchmark_matmul.py in the VS Code Colab runtime.

Usage:
  ./benchmarks/run_matmul_colab.sh [benchmark_matmul.py arguments]

With no arguments, benchmark the best Version 22 BF16 tile found on A100:
  --sizes 4096 8192 --tiles 64x16x64 --dtype bf16 --warmup 3 --repeats 10

Examples:
  ./benchmarks/run_matmul_colab.sh
  ./benchmarks/run_matmul_colab.sh \
      --sizes 1024 \
      --tiles 32x16x32 \
      --dtype bf16 \
      --warmup 2 \
      --repeats 5 \
      --no-graphs

Environment variables:
  MYTRITON_COLAB_ROOT      Remote project path (default: /content/mytriton)
  MYTRITON_COLAB_TIMEOUT   Timeout in seconds (default: 1800)
  COLABRUN_NOTEBOOK        Local notebook used by colabrun
                           (default: tests/mytriton_colab_tests.ipynb)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v colabrun >/dev/null 2>&1; then
    echo "error: colabrun is not installed or is not on PATH" >&2
    exit 1
fi

TEMP_DIR=$(mktemp -d /tmp/mytriton-colab-benchmark.XXXXXX)
NOTEBOOK_BACKUP="$TEMP_DIR/colabrun-notebook.ipynb"

if [[ -f "$COLAB_NOTEBOOK" ]]; then
    cp -- "$COLAB_NOTEBOOK" "$NOTEBOOK_BACKUP"
fi

cleanup() {
    if [[ -f "$NOTEBOOK_BACKUP" ]]; then
        cp -- "$NOTEBOOK_BACKUP" "$COLAB_NOTEBOOK"
    fi
    rm -rf -- "$TEMP_DIR"
}

trap cleanup EXIT

if [[ $# -eq 0 ]]; then
    set -- \
        --sizes 4096 8192 \
        --tiles 64x16x64 \
        --dtype bf16 \
        --warmup 3 \
        --repeats 10 \
        --artifacts-dir benchmarks/artifacts/colab-benchmark \
        --output benchmarks/artifacts/colab-benchmark.json
fi

COLABRUN=(
    colabrun
    --timeout "$COLAB_TIMEOUT"
    --cwd "$REMOTE_ROOT"
    --notebook "$COLAB_NOTEBOOK"
)

echo "Running benchmark on Colab from $REMOTE_ROOT"
"${COLABRUN[@]}" python benchmarks/benchmark_matmul.py "$@"
