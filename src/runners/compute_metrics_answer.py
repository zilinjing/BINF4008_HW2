import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CheXpert label metrics and report-level BERTScore."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for this evaluation run.",
    )
    return parser.parse_args()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _is_placeholder(value: Any) -> bool:
    text = str(value).strip()
    if text == "":
        return True
    if text.startswith("<FILL_ME_"):
        return True
    if text.startswith("<OPTIONAL_FILL_ME_"):
        return True
    return False


def _resolve_required_path(value: Any, field_name: str) -> Path:
    if _is_placeholder(value):
        raise ValueError(
            f"Config field '{field_name}' is not filled. "
            f"Please replace the placeholder with a real path."
        )
    return Path(str(value).strip()).expanduser()


def _resolve_optional_path(value: Any) -> Optional[Path]:
    if value is None or _is_placeholder(value):
        return None
    return Path(str(value).strip()).expanduser()


def _normalize_path(path: str) -> str:
    norm = str(path).strip()
    marker = "CheXpert-v1.0/"
    if marker in norm:
        norm = norm.split(marker, 1)[1]
    return norm.lstrip("/")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _binary_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric == 1).astype(int)


def _compute_tpr(df: pd.DataFrame, label: str) -> Optional[float]:
    gt_col = f"gt_{label}"
    pred_col = f"pred_{label}"
    if gt_col not in df.columns or pred_col not in df.columns:
        return None
    gt = _binary_series(df[gt_col])
    pred = _binary_series(df[pred_col])
    positives = gt == 1
    denom = int(positives.sum())
    if denom == 0:
        return None
    tp = int(((pred == 1) & positives).sum())
    return float(tp / denom)


def _compute_prf(gt: pd.Series, pred: pd.Series) -> Dict[str, Optional[float]]:
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    support = int((gt == 1).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = float((2 * precision * recall) / (precision + recall))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _is_empty_text(value: Any) -> bool:
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan"}


def _compute_bertscore(
    output_dir: Path,
    config: Dict[str, Any],
    predictions_path: Optional[Path] = None,
) -> None:
    if predictions_path is None:
        predictions_path = output_dir / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"[INFO] Predictions file missing at {predictions_path}; skipping BERTScore.")
        return

    rows: List[Dict[str, Any]] = []
    with predictions_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            impressions = record.get("impressions", "")
            prediction = record.get("prediction", "")
            if _is_empty_text(impressions) or _is_empty_text(prediction):
                continue
            rows.append(
                {
                    "row_id": record.get("row_id"),
                    "impressions": str(impressions),
                    "prediction": str(prediction),
                }
            )

    if not rows:
        print("[INFO] No valid (impressions, prediction) pairs found; skipping BERTScore.")
        return

    metrics_cfg = config.get("metrics", {})
    bert_cfg = metrics_cfg.get("bertscore", {})
    model_type = bert_cfg.get("model_type", "distilbert-base-uncased")
    lang = bert_cfg.get("lang", "en")
    batch_size = int(bert_cfg.get("batch_size", 16))
    device = bert_cfg.get("device", None)
    rescale_with_baseline = bool(bert_cfg.get("rescale_with_baseline", False))

    try:
        from bert_score import score as bert_score_score
    except ImportError:
        print("[WARN] `bert_score` is not installed; skipping BERTScore.")
        return

    refs = [r["impressions"] for r in rows]
    cands = [r["prediction"] for r in rows]

    try:
        precision, recall, f1 = bert_score_score(
            cands=cands,
            refs=refs,
            model_type=model_type,
            lang=lang,
            batch_size=batch_size,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=True,
        )
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
        return

    per_row_records = []
    for i, base in enumerate(rows):
        per_row_records.append(
            {
                "row_id": base["row_id"],
                "impressions": base["impressions"],
                "prediction": base["prediction"],
                "bertscore_precision": float(precision[i].item()),
                "bertscore_recall": float(recall[i].item()),
                "bertscore_f1": float(f1[i].item()),
            }
        )
    per_row_df = pd.DataFrame(per_row_records)
    per_row_df.to_csv(output_dir / "metrics_bertscore_per_row.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "n_pairs": len(rows),
                "model_type": model_type,
                "bertscore_precision_mean": float(precision.mean().item()),
                "bertscore_recall_mean": float(recall.mean().item()),
                "bertscore_f1_mean": float(f1.mean().item()),
            }
        ]
    )
    summary_df.to_csv(output_dir / "metrics_bertscore_summary.csv", index=False)


def _compute_label_metrics(
    labels_and_predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Homework focus:
    - Call your `_compute_prf` implementation per disease label.
    - Finish summary aggregation (macro/micro F1) below.
    """
    if "relative_path" not in labels_and_predictions.columns and "image_path" in labels_and_predictions.columns:
        labels_and_predictions["relative_path"] = labels_and_predictions["image_path"].map(_normalize_path)

    f1_rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for label in CHEXPERT_LABELS:
        gt_col = f"gt_{label}"
        pred_col = f"pred_{label}"
        if gt_col not in labels_and_predictions.columns:
            continue
        if pred_col not in labels_and_predictions.columns:
            continue

        gt = _binary_series(labels_and_predictions[gt_col])
        pred = _binary_series(labels_and_predictions[pred_col])
        prf = _compute_prf(gt, pred)
        total_tp += prf["tp"]
        total_fp += prf["fp"]
        total_fn += prf["fn"]

        f1_rows.append(
            {
                "label": label,
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "support": prf["support"],
                "tp": prf["tp"],
                "fp": prf["fp"],
                "fn": prf["fn"],
            }
        )

    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv(output_dir / "metrics_f1_by_disease.csv", index=False)

    valid_f1 = [row["f1"] for row in f1_rows if row["f1"] is not None]
    macro_f1 = float(sum(valid_f1) / len(valid_f1)) if valid_f1 else None
    micro_precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else None
    micro_recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else None
    if micro_precision is None or micro_recall is None or (micro_precision + micro_recall) == 0:
        micro_f1 = None
    else:
        micro_f1 = float((2 * micro_precision * micro_recall) / (micro_precision + micro_recall))

    summary_df = pd.DataFrame(
        [
            {
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "num_labels": len(f1_rows),
            }
        ]
    )
    summary_df.to_csv(output_dir / "metrics_f1_summary.csv", index=False)

    # Keep original TPR outputs for backward compatibility.
    overall_rows = []
    for label in CHEXPERT_LABELS:
        if f"gt_{label}" not in labels_and_predictions.columns:
            continue
        if f"pred_{label}" not in labels_and_predictions.columns:
            continue
        overall_rows.append(
            {
                "label": label,
                "tpr": _compute_tpr(labels_and_predictions, label),
            }
        )
    pd.DataFrame(overall_rows).to_csv(output_dir / "metrics_tpr_overall.csv", index=False)


def _compute_subgroup_tpr(
    labels_and_predictions: pd.DataFrame,
    output_dir: Path,
    splits_path: Union[str, Path, None],
) -> None:
    if not splits_path:
        print("[INFO] No splits path provided; skipping race/gender TPR.")
        return

    splits_path = Path(splits_path)
    if not splits_path.exists():
        print(f"[INFO] Splits not found at {splits_path}; skipping race/gender TPR.")
        return

    metadata_df = pd.read_csv(splits_path)
    path_col = _find_column(
        metadata_df,
        ["path_to_image", "path", "image_path", "image", "Path"],
    )
    if not path_col:
        print("[INFO] Metadata missing path column; skipping race/gender TPR.")
        return

    if "relative_path" not in labels_and_predictions.columns:
        if "image_path" not in labels_and_predictions.columns:
            print("[INFO] labels.csv missing image_path; skipping race/gender TPR.")
            return
        labels_and_predictions["relative_path"] = labels_and_predictions["image_path"].map(_normalize_path)

    relative_series = metadata_df[path_col].map(_normalize_path).rename("relative_path")
    metadata_df = pd.concat([metadata_df, relative_series], axis=1)
    race_col = _find_column(metadata_df, ["race", "Race", "RACE", "patient_race"])
    gender_col = _find_column(metadata_df, ["gender", "Gender", "sex", "Sex", "SEX", "patient_gender"])
    race_onehot_cols = [c for c in metadata_df.columns if c.lower().startswith("race_")]
    gender_onehot_cols = [
        c for c in metadata_df.columns
        if c.lower().startswith("gender_") or c.lower().startswith("sex_")
    ]

    if not race_col and not gender_col and not race_onehot_cols and not gender_onehot_cols:
        print("[INFO] Metadata missing race/gender columns; skipping subgroup TPR.")
        return

    merge_cols = ["relative_path"]
    merge_cols += [c for c in [race_col, gender_col] if c]
    merge_cols += race_onehot_cols + gender_onehot_cols
    merged = labels_and_predictions.merge(
        metadata_df[merge_cols],
        on="relative_path",
        how="left",
    )

    if "Frontal/Lateral_Lateral" in merged.columns:
        merged = merged[merged["Frontal/Lateral_Lateral"] != 1]

    if race_col:
        merged[race_col] = merged[race_col].fillna("Unknown")
    if gender_col:
        merged[gender_col] = merged[gender_col].fillna("Unknown")
    for col in race_onehot_cols + gender_onehot_cols:
        merged[col] = merged[col].fillna(0)

    def _add_group_rows(
        group_df: pd.DataFrame,
        group_type: str,
        group_value: str,
        output_rows: List[Dict[str, Any]],
    ) -> None:
        for label in CHEXPERT_LABELS:
            if f"gt_{label}" not in group_df.columns:
                continue
            if f"pred_{label}" not in group_df.columns:
                continue
            output_rows.append(
                {
                    "group_type": group_type,
                    "group": group_value,
                    "label": label,
                    "tpr": _compute_tpr(group_df, label),
                    "positives": int((_binary_series(group_df[f"gt_{label}"]) == 1).sum()),
                }
            )

    race_rows: List[Dict[str, Any]] = []
    gender_rows: List[Dict[str, Any]] = []

    def _normalize_race_value(value: str) -> str:
        lowered = str(value).strip().lower()
        if lowered in {"white", "caucasian"}:
            return "white"
        if lowered in {"black", "african", "african american", "african-american"}:
            return "black"
        if lowered in {"asian"}:
            return "asian"
        return "other"

    if race_onehot_cols:
        race_groups: Dict[str, List[str]] = {"white": [], "black": [], "asian": [], "other": []}
        for col in race_onehot_cols:
            raw_value = col.split("_", 1)[1] if "_" in col else col
            bucket = _normalize_race_value(raw_value)
            race_groups[bucket].append(col)
        for group_value, cols in race_groups.items():
            if not cols:
                continue
            group_df = merged[merged[cols].fillna(0).sum(axis=1) > 0]
            _add_group_rows(group_df, "race", group_value, race_rows)
    elif race_col:
        merged["race_group"] = merged[race_col].map(_normalize_race_value)
        for value, group_df in merged.groupby("race_group", dropna=False):
            _add_group_rows(group_df, "race", str(value), race_rows)

    if gender_onehot_cols:
        for col in gender_onehot_cols:
            group_value = col.split("_", 1)[1] if "_" in col else col
            group_df = merged[merged[col] == 1]
            _add_group_rows(group_df, "gender", group_value, gender_rows)
    elif gender_col:
        for value, group_df in merged.groupby(gender_col, dropna=False):
            _add_group_rows(group_df, "gender", str(value), gender_rows)

    if race_rows:
        race_df = pd.DataFrame(race_rows)
        race_df.to_csv(output_dir / "metrics_tpr_by_race.csv", index=False)
        race_pivot = race_df.pivot_table(
            index="label",
            columns="group",
            values="tpr",
            aggfunc="first",
        ).reset_index()
        race_pivot.to_latex(output_dir / "metrics_tpr_by_race.tex", index=False, float_format="%.3f")
    if gender_rows:
        pd.DataFrame(gender_rows).to_csv(output_dir / "metrics_tpr_by_gender.csv", index=False)


def compute_metrics(config: Dict[str, Any]) -> None:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    metrics_cfg = config.get("metrics", {})

    dataset_name = dataset_cfg["name"]
    model_name = model_cfg["name"]

    output_dir_cfg = metrics_cfg.get("output_dir", f"experiments/{dataset_name}/{model_name}/")
    output_dir = _resolve_required_path(output_dir_cfg, "metrics.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path_cfg = metrics_cfg.get("labels_path", config.get("labels_path"))
    labels_path = (
        _resolve_required_path(labels_path_cfg, "metrics.labels_path")
        if labels_path_cfg
        else (output_dir / "labels.csv")
    )
    if labels_path.exists():
        labels_and_predictions = pd.read_csv(labels_path)
        _compute_label_metrics(labels_and_predictions, output_dir)
        _compute_subgroup_tpr(
            labels_and_predictions,
            output_dir,
            _resolve_optional_path(config.get("splits")),
        )
    else:
        print(f"[INFO] Labels not generated at {labels_path}; skipping label-based metrics.")

    predictions_path_cfg = metrics_cfg.get("predictions_path", config.get("predictions_path"))
    predictions_path = (
        _resolve_required_path(predictions_path_cfg, "metrics.predictions_path")
        if predictions_path_cfg
        else None
    )
    _compute_bertscore(output_dir, config, predictions_path=predictions_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    compute_metrics(cfg)


if __name__ == "__main__":
    main()
