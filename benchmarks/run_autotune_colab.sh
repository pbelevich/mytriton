#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

"$REPO_ROOT/tools/sync_colab.sh"

COLABRUN=${COLABRUN:-$HOME/.local/bin/colabrun}
NOTEBOOK=${MYTRITON_COLAB_NOTEBOOK:-$REPO_ROOT/tests/mytriton_colab_tests.ipynb}

"$COLABRUN" \
    --notebook "$NOTEBOOK" \
    --timeout 1800 \
    --cwd /content/mytriton \
    python benchmarks/benchmark_autotune.py "$@"
