#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

REMOTE_ROOT=${MYTRITON_COLAB_ROOT:-/content/mytriton}
COLAB_TIMEOUT=${MYTRITON_COLAB_TIMEOUT:-1800}
COLAB_NOTEBOOK=${COLABRUN_NOTEBOOK:-$REPO_ROOT/tests/mytriton_colab_tests.ipynb}
REMOTE_OUTPUT=${1:-benchmarks/artifacts/colab-profile}

usage() {
    cat <<'EOF'
Run the reproducible A100 benchmark and Nsight Compute capture on Colab.

Usage:
  ./benchmarks/profile_matmul_colab.sh [remote-output-directory]

The underlying profile_matmul_a100.sh captures generated CUDA, PTX, exact
NVRTC cubin, SASS, resource usage, an .ncu-rep file, and text profiler output.

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

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

if ! command -v colabrun >/dev/null 2>&1; then
    echo "error: colabrun is not installed or is not on PATH" >&2
    exit 1
fi

TEMP_DIR=$(mktemp -d /tmp/mytriton-colab-profile.XXXXXX)
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

COLABRUN=(
    colabrun
    --timeout "$COLAB_TIMEOUT"
    --cwd "$REMOTE_ROOT"
    --notebook "$COLAB_NOTEBOOK"
)

echo "Profiling mytriton on Colab; remote output: $REMOTE_OUTPUT"
"${COLABRUN[@]}" ./benchmarks/profile_matmul_a100.sh "$REMOTE_OUTPUT"
