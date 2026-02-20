#!/usr/bin/env bash
set -euo pipefail

# Activate uv venv
source cxr_reason/bin/activate

export VLLM_USE_TRITON_FLASH_ATTN=1

# Run CheXpert eval from the repo root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

python -m src.runners.run_eval --config configs/chexpert.yaml