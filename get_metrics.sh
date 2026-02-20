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
EXP_DIR="experiments/${DATASET_NAME}/${MODEL_NAME}"
mkdir -p "$EXP_DIR"

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
PREDICTIONS_SRC="${PREDICTIONS_PATH:-${CONFIG_DIR}/predictions.jsonl}"
PREDICTIONS_DST="${EXP_DIR}/predictions.jsonl"
if [[ -f "$PREDICTIONS_SRC" ]]; then
  cp "$PREDICTIONS_SRC" "$PREDICTIONS_DST"
  echo "[INFO] Copied predictions to ${PREDICTIONS_DST} for BERTScore."
elif [[ -f "$PREDICTIONS_DST" ]]; then
  echo "[INFO] Using existing predictions at ${PREDICTIONS_DST}."
else
  echo "[ERROR] predictions.jsonl not found."
  echo "        Checked: ${PREDICTIONS_SRC} and ${PREDICTIONS_DST}"
  exit 1
fi

LABELS_PATH="${LABELS_PATH:-${EXP_DIR}/labels.csv}"
if [[ ! -f "$LABELS_PATH" ]]; then
  echo "[ERROR] labels.csv not found at ${LABELS_PATH}."
  echo "        Run ./extract_labels.sh ${CONFIG_PATH} first."
  exit 1
fi

if ! python - <<'PY'
import importlib.util
import sys
missing = [m for m in ("bert_score", "torch", "transformers") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
then
  echo "[ERROR] Missing Python deps for BERTScore."
  echo "        Please install: bert-score torch transformers"
  exit 1
fi

echo "[INFO] Running metrics with config: $CONFIG_PATH"
python -m src.runners.compute_metrics --config "$CONFIG_PATH"

echo "[INFO] Generated metric files in ${EXP_DIR}:"
ls -1 "${EXP_DIR}"/metrics_* 2>/dev/null || true
