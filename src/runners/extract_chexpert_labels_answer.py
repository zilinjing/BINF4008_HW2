import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

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

REL_PATH_PATTERN = re.compile(
    r"(train|valid|test)/patient\d+/study\d+/view[^/\s]+",
    flags=re.IGNORECASE,
)
PATIENT_PATTERN = re.compile(r"patient\d+", flags=re.IGNORECASE)
STUDY_PATTERN = re.compile(r"study\d+", flags=re.IGNORECASE)
VIEW_PATTERN = re.compile(r"view[^/\s]+", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CheXpert labels from generated reports using a Transformers LLM."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.jsonl.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to CheXpert CSV containing path and CheXpert labels.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to write extracted labels CSV.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Transformers model ID used to extract labels from generated reports.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max generated tokens per report for JSON labels.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k for sampling. Use -1 to disable.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for Transformers model loading.",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include impressions/prediction text columns in output CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of prediction rows.",
    )

    # Backward-compatible args kept as no-ops so existing scripts do not break.
    parser.add_argument("--snapshot", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gt-labels-csv", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--refresh-gt-labels", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-model-len", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def normalize_image_path(path: str) -> str:
    if path is None:
        return ""
    norm = str(path).strip().replace("\\", "/")
    rel_match = REL_PATH_PATTERN.search(norm)
    if rel_match:
        return rel_match.group(0)
    marker = "CheXpert-v1.0/"
    if marker in norm:
        norm = norm.split(marker, 1)[1]
    return norm.lstrip("/")


def _extract_token(path: str, pattern: re.Pattern) -> str:
    if path is None:
        return ""
    match = pattern.search(str(path))
    return match.group(0).lower() if match else ""


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_label_json(text: str) -> Dict[str, int]:
    cleaned = str(text).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        json_blob = _extract_json_object(cleaned)
        if not json_blob:
            return {}
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    output: Dict[str, int] = {}
    for label, value in parsed.items():
        try:
            output[label] = 1 if int(value) == 1 else 0
        except (ValueError, TypeError):
            output[label] = 0
    return output


def _normalize_label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def coerce_labels(label_dict: Dict[str, int]) -> Dict[str, int]:
    lookup = {_normalize_label_key(label): label for label in CHEXPERT_LABELS}
    coerced = {label: 0 for label in CHEXPERT_LABELS}
    for key, value in label_dict.items():
        canonical = lookup.get(_normalize_label_key(key))
        if canonical is None:
            continue
        try:
            coerced[canonical] = 1 if int(value) == 1 else 0
        except (TypeError, ValueError):
            coerced[canonical] = 0
    return coerced


def _binary_value(value: Any) -> int:
    try:
        return 1 if int(float(value)) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _resolve_gt_label_columns(df: pd.DataFrame) -> Dict[str, str]:
    label_cols: Dict[str, str] = {}
    missing: List[str] = []
    for label in CHEXPERT_LABELS:
        if label in df.columns:
            label_cols[label] = label
            continue
        underscore = label.replace(" ", "_")
        if underscore in df.columns:
            label_cols[label] = underscore
            continue
        missing.append(label)
    if missing:
        raise ValueError(
            f"Ground truth CSV missing required label columns: {missing}"
        )
    return label_cols


def load_ground_truth_rows(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    path_col = _find_column(df, ["Path", "path_to_image", "path", "image_path", "image"])
    if not path_col:
        raise ValueError("Ground truth CSV missing path column (e.g., 'Path').")
    label_cols = _resolve_gt_label_columns(df)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_path = str(row[path_col])
        rows.append(
            {
                "gt_path": raw_path,
                "relative_path": normalize_image_path(raw_path),
                "patient": _extract_token(raw_path, PATIENT_PATTERN),
                "study": _extract_token(raw_path, STUDY_PATTERN),
                "view": _extract_token(raw_path, VIEW_PATTERN),
                "labels": {
                    label: _binary_value(row[label_cols[label]])
                    for label in CHEXPERT_LABELS
                },
            }
        )
    return rows


def load_prediction_rows(path: Path, max_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as fin:
        for line in fin:
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = str(record.get("image_path", ""))
            rows.append(
                {
                    "row_id": record.get("row_id", len(rows)),
                    "image_path": image_path,
                    "relative_path": normalize_image_path(image_path),
                    "patient": _extract_token(image_path, PATIENT_PATTERN),
                    "study": _extract_token(image_path, STUDY_PATTERN),
                    "view": _extract_token(image_path, VIEW_PATTERN),
                    "impressions": str(record.get("impressions", "")),
                    "prediction": str(record.get("prediction", "")),
                }
            )
    return rows


def _build_index(
    rows: List[Dict[str, Any]],
    key_fn,
) -> Dict[Any, List[int]]:
    index: Dict[Any, List[int]] = {}
    for i, row in enumerate(rows):
        key = key_fn(row)
        if not key:
            continue
        index.setdefault(key, []).append(i)
    return index


def _take_first_unused(candidates: List[int], used: set[int]) -> Optional[int]:
    for idx in candidates:
        if idx not in used:
            used.add(idx)
            return idx
    return None


def map_prediction_to_gt(
    pred_rows: List[Dict[str, Any]],
    gt_rows: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int, str]], Dict[str, int]]:
    by_relative = _build_index(gt_rows, lambda r: r["relative_path"])
    by_psv = _build_index(gt_rows, lambda r: (r["patient"], r["study"], r["view"]))
    by_ps = _build_index(gt_rows, lambda r: (r["patient"], r["study"]))
    by_patient = _build_index(gt_rows, lambda r: r["patient"])

    used_gt: set[int] = set()
    matches: List[Tuple[int, int, str]] = []
    stats = {
        "relative_path": 0,
        "patient_study_view": 0,
        "patient_study": 0,
        "patient_only": 0,
        "row_order_fallback": 0,
        "unmatched": 0,
    }

    for pred_idx, pred_row in enumerate(pred_rows):
        gt_idx = None
        method = ""

        rel_key = pred_row["relative_path"]
        if rel_key:
            gt_idx = _take_first_unused(by_relative.get(rel_key, []), used_gt)
            if gt_idx is not None:
                method = "relative_path"

        if gt_idx is None and pred_row["patient"] and pred_row["study"] and pred_row["view"]:
            key = (pred_row["patient"], pred_row["study"], pred_row["view"])
            gt_idx = _take_first_unused(by_psv.get(key, []), used_gt)
            if gt_idx is not None:
                method = "patient_study_view"

        if gt_idx is None and pred_row["patient"] and pred_row["study"]:
            key = (pred_row["patient"], pred_row["study"])
            gt_idx = _take_first_unused(by_ps.get(key, []), used_gt)
            if gt_idx is not None:
                method = "patient_study"

        if gt_idx is None and pred_row["patient"]:
            gt_idx = _take_first_unused(by_patient.get(pred_row["patient"], []), used_gt)
            if gt_idx is not None:
                method = "patient_only"

        if gt_idx is None:
            if pred_idx < len(gt_rows) and pred_idx not in used_gt:
                gt_idx = pred_idx
                used_gt.add(gt_idx)
                method = "row_order_fallback"
            else:
                for candidate in range(len(gt_rows)):
                    if candidate not in used_gt:
                        gt_idx = candidate
                        used_gt.add(gt_idx)
                        method = "row_order_fallback"
                        break

        if gt_idx is None:
            stats["unmatched"] += 1
            continue

        stats[method] += 1
        matches.append((pred_idx, gt_idx, method))

    return matches, stats


def build_prompt(report_text: str) -> str:
    labels = ", ".join(CHEXPERT_LABELS)
    return (
        "Extract CheXpert labels from this report. "
        "Output valid JSON only with keys exactly: "
        f"{labels}. "
        "For each label, use 1 if present, otherwise 0. "
        "Use 0 for uncertain or not mentioned findings.\n\n"
        "Report:\n"
        f"{report_text}\n\n"
        "JSON:"
    )


def resolve_dtype(dtype: str, torch_module):
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float32":
        return torch_module.float32
    return "auto"


def load_transformers_model(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Missing dependency for label extraction. "
            "Install `transformers`, `accelerate`, and `sentencepiece`."
        ) from e

    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = resolve_dtype(args.dtype, torch)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )
    model.eval()
    return model, tokenizer, torch


def _format_chat_prompt(tokenizer, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": "You are an expert radiology label extractor. Return only JSON.",
            },
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


def batch_extract(
    model,
    tokenizer,
    torch_module,
    prompts: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, int]]:
    formatted_prompts = [_format_chat_prompt(tokenizer, p) for p in prompts]
    tokenized = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    try:
        model_device = next(model.parameters()).device
        tokenized = {k: v.to(model_device) for k, v in tokenized.items()}
    except StopIteration:
        pass

    do_sample = args.temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
    if args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k
    if not do_sample:
        # Force greedy-compatible defaults to avoid sampling-flag warnings
        # from model generation configs that carry non-default sampling params.
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 50

    with torch_module.inference_mode():
        output_ids = model.generate(**tokenized, **gen_kwargs)

    prompt_len = tokenized["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    extracted = []
    for text in generated_texts:
        extracted.append(coerce_labels(parse_label_json(text)))
    return extracted


def write_rows(
    writer: csv.DictWriter,
    rows: Iterable[Dict[str, Any]],
) -> None:
    for row in rows:
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_rows = load_prediction_rows(predictions_path, args.max_rows)
    gt_rows = load_ground_truth_rows(ground_truth_path)

    if not pred_rows:
        raise ValueError(f"No prediction rows found in {predictions_path}")
    if not gt_rows:
        raise ValueError(f"No ground-truth rows found in {ground_truth_path}")

    if len(pred_rows) != len(gt_rows):
        print(
            "[WARN] Prediction/ground-truth row counts differ: "
            f"{len(pred_rows)} vs {len(gt_rows)}"
        )

    matches, map_stats = map_prediction_to_gt(pred_rows, gt_rows)
    print("[INFO] Mapping summary:", map_stats)

    model, tokenizer, torch_module = load_transformers_model(args)

    base_fields = [
        "row_id",
        "image_path",
        "relative_path",
        "gt_path",
        "match_method",
    ]
    if args.include_text:
        base_fields += ["impressions", "pred_report"]
    gt_fields = [f"gt_{label}" for label in CHEXPERT_LABELS]
    pred_fields = [f"pred_{label}" for label in CHEXPERT_LABELS]
    fieldnames = base_fields + gt_fields + pred_fields

    with output_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for start in range(0, len(matches), args.batch_size):
            batch = matches[start : start + args.batch_size]
            prompts = [build_prompt(pred_rows[p]["prediction"]) for p, _, _ in batch]
            pred_label_sets = batch_extract(
                model=model,
                tokenizer=tokenizer,
                torch_module=torch_module,
                prompts=prompts,
                args=args,
            )

            output_rows: List[Dict[str, Any]] = []
            for i, (pred_idx, gt_idx, match_method) in enumerate(batch):
                pred_row = pred_rows[pred_idx]
                gt_row = gt_rows[gt_idx]
                output_row = {
                    "row_id": pred_row["row_id"],
                    "image_path": pred_row["image_path"],
                    "relative_path": pred_row["relative_path"],
                    "gt_path": gt_row["gt_path"],
                    "match_method": match_method,
                }
                if args.include_text:
                    output_row["impressions"] = pred_row["impressions"]
                    output_row["pred_report"] = pred_row["prediction"]
                gt_labels = gt_row["labels"]
                pred_labels = pred_label_sets[i]
                for label in CHEXPERT_LABELS:
                    output_row[f"gt_{label}"] = gt_labels[label]
                    output_row[f"pred_{label}"] = pred_labels[label]
                output_rows.append(output_row)

            write_rows(writer, output_rows)


if __name__ == "__main__":
    main()
