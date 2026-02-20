#!/usr/bin/env bash
set -euo pipefail

if [[ -f "cxr_reason/bin/activate" ]]; then
  source cxr_reason/bin/activate
else
  echo "[WARN] Virtualenv activation script not found at cxr_reason/bin/activate; using current Python."
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/chexpert.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH"
  exit 1
fi

DATASET_NAME="$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg.get("dataset", {}).get("name", "chexpert"))
PY
)"

MODEL_NAME="$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg.get("model", {}).get("name", "model"))
PY
)"

BASE_DIR="$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg.get("dataset", {}).get("base_dir", ""))
PY
)"

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
EXP_DIR="experiments/${DATASET_NAME}/${MODEL_NAME}"
mkdir -p "$EXP_DIR"

PREDICTIONS_PATH="${PREDICTIONS_PATH:-${CONFIG_DIR}/predictions.jsonl}"
if [[ ! -f "$PREDICTIONS_PATH" ]]; then
  FALLBACK_PRED="experiments/${DATASET_NAME}/${MODEL_NAME}/predictions.jsonl"
  if [[ -f "$FALLBACK_PRED" ]]; then
    PREDICTIONS_PATH="$FALLBACK_PRED"
  fi
fi
if [[ ! -f "$PREDICTIONS_PATH" ]]; then
  echo "[ERROR] predictions.jsonl not found."
  echo "        Checked: ${CONFIG_DIR}/predictions.jsonl and experiments/${DATASET_NAME}/${MODEL_NAME}/predictions.jsonl"
  exit 1
fi

if [[ -n "${GROUND_TRUTH_PATH:-}" ]]; then
  GT_PATH="$GROUND_TRUTH_PATH"
elif [[ -n "$BASE_DIR" && -f "$BASE_DIR/valid.csv" ]]; then
  GT_PATH="$BASE_DIR/valid.csv"
elif [[ -n "$BASE_DIR" && -f "$BASE_DIR/train.csv" ]]; then
  GT_PATH="$BASE_DIR/train.csv"
else
  echo "[ERROR] Ground-truth CSV not found."
  echo "        Set GROUND_TRUTH_PATH or ensure valid.csv/train.csv exists under dataset.base_dir."
  exit 1
fi

LABELS_OUT="${LABELS_OUT:-${EXP_DIR}/labels.csv}"
MODEL_ARG=()
if [[ -n "${LABEL_EXTRACT_MODEL_ID:-}" ]]; then
  MODEL_ARG=(--model-id "$LABEL_EXTRACT_MODEL_ID")
fi
GPU_ARG=()
if [[ -n "${GPU_DEVICES:-}" ]]; then
  GPU_ARG=(--gpu-devices "$GPU_DEVICES")
fi
BATCH_SIZE_ARG=()
if [[ -n "${LABEL_BATCH_SIZE:-}" ]]; then
  BATCH_SIZE_ARG=(--batch-size "$LABEL_BATCH_SIZE")
fi

if ! python - <<'PY'
import importlib.util
import sys
missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
then
  echo "[ERROR] Missing Python deps for label extraction."
  echo "        Please install: torch transformers accelerate sentencepiece"
  exit 1
fi

echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Predictions: $PREDICTIONS_PATH"
echo "[INFO] Ground truth: $GT_PATH"
echo "[INFO] Output labels: $LABELS_OUT"

python -m src.runners.extract_chexpert_labels \
  --predictions "$PREDICTIONS_PATH" \
  --ground-truth "$GT_PATH" \
  --output-csv "$LABELS_OUT" \
  --include-text \
  "${MODEL_ARG[@]}" \
  "${GPU_ARG[@]}" \
  "${BATCH_SIZE_ARG[@]}"
