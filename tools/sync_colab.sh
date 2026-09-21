#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

REMOTE_ROOT=${MYTRITON_COLAB_ROOT:-/content/mytriton}
COLAB_TIMEOUT=${MYTRITON_COLAB_TIMEOUT:-600}
COLAB_EXTRAS=${MYTRITON_COLAB_EXTRAS:-cuda12}
COLAB_NOTEBOOK=${COLABRUN_NOTEBOOK:-$REPO_ROOT/tests/mytriton_colab_tests.ipynb}
CHUNK_BYTES=48000

usage() {
    cat <<'EOF'
Upload the current mytriton checkout to the VS Code Colab runtime and install it.

Usage:
  ./tools/sync_colab.sh

Environment variables:
  MYTRITON_COLAB_ROOT      Remote project path (default: /content/mytriton)
  MYTRITON_COLAB_TIMEOUT   Timeout in seconds (default: 600)
  MYTRITON_COLAB_EXTRAS    pip extra to install (default: cuda12)
  COLABRUN_NOTEBOOK        Local notebook used by colabrun
                           (default: tests/mytriton_colab_tests.ipynb)

The script uploads the working tree itself, including uncommitted changes. It
does not require a commit or a GitHub push.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

if ! command -v colabrun >/dev/null 2>&1; then
    echo "error: colabrun is not installed or is not on PATH" >&2
    exit 1
fi

if [[ ! "$REMOTE_ROOT" =~ ^/content/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
    echo "error: MYTRITON_COLAB_ROOT must be a concrete path below /content" >&2
    exit 1
fi

COLABRUN=(
    colabrun
    --timeout "$COLAB_TIMEOUT"
    --notebook "$COLAB_NOTEBOOK"
)

run_colab() {
    "${COLABRUN[@]}" "$@"
}

TEMP_DIR=$(mktemp -d /tmp/mytriton-colab-sync.XXXXXX)
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

ARCHIVE="$TEMP_DIR/mytriton.tar.gz"
PARTS_DIR="$TEMP_DIR/parts"
mkdir -p "$PARTS_DIR"

echo "Packing current checkout from $REPO_ROOT"
tar \
    --create \
    --gzip \
    --file "$ARCHIVE" \
    --directory "$REPO_ROOT" \
    --exclude='.git' \
    --exclude='.mypy_cache' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='benchmarks/artifacts' \
    .

ARCHIVE_SHA256=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
ARCHIVE_SIZE=$(wc -c < "$ARCHIVE")
split --bytes "$CHUNK_BYTES" "$ARCHIVE" "$PARTS_DIR/part-"

REMOTE_ARCHIVE=/content/mytriton-upload.tar.gz

echo "Uploading $ARCHIVE_SIZE bytes to the Colab runtime"
run_colab python -c \
    'from pathlib import Path; Path("/content/mytriton-upload.tar.gz").write_bytes(b"")'

part_count=0
for part in "$PARTS_DIR"/part-*; do
    encoded=$(base64 --wrap=0 "$part")
    run_colab python -c \
        'import base64, sys; from pathlib import Path; path = Path(sys.argv[1]); path.open("ab").write(base64.b64decode(sys.argv[2]))' \
        "$REMOTE_ARCHIVE" \
        "$encoded"
    part_count=$((part_count + 1))
done

echo "Uploaded $part_count chunks; verifying and extracting"
run_colab python -c \
    'import hashlib, shutil, sys, tarfile; from pathlib import Path; archive = Path(sys.argv[1]); root = Path(sys.argv[2]); expected = sys.argv[3]; actual = hashlib.sha256(archive.read_bytes()).hexdigest(); assert actual == expected, f"archive checksum mismatch: {actual} != {expected}"; shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True); tarfile.open(archive).extractall(root, filter="data"); print(f"extracted {archive.stat().st_size} bytes to {root} ({actual})")' \
    "$REMOTE_ARCHIVE" \
    "$REMOTE_ROOT" \
    "$ARCHIVE_SHA256"

if [[ -n "$COLAB_EXTRAS" ]]; then
    INSTALL_TARGET="$REMOTE_ROOT[$COLAB_EXTRAS]"
else
    INSTALL_TARGET="$REMOTE_ROOT"
fi

echo "Installing $INSTALL_TARGET in editable mode"
run_colab python -m pip install --quiet --editable "$INSTALL_TARGET"

run_colab python -c \
    'from pathlib import Path; import mytriton; print(f"mytriton: {Path(mytriton.__file__).resolve()}"); import torch; print(f"torch: {torch.__version__}"); print(f"CUDA: {torch.version.cuda}"); print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else None}")'

echo
echo "Current checkout is ready at $REMOTE_ROOT"
echo "Run: ./benchmarks/run_matmul_colab.sh"
