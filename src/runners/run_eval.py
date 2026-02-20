import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
import yaml
import warnings
import logging

from tqdm import tqdm

from src.data.clean import get_cleaned
from src.models.__init__ import build_model_from_config

warnings.filterwarnings("ignore")

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.core").setLevel(logging.ERROR)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot VQA evaluation on chest X-rays."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for this evaluation run.",
    )
    return parser.parse_args()

def load_config(config_path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_config_copy(config: Dict[str, Any], out_dir: Path) -> None:
    """Save a snapshot of the config used for this run."""
    config_copy_path = out_dir / "config_used.yaml"
    with config_copy_path.open("w") as f:
        yaml.safe_dump(config, f)


def _is_placeholder(value: Any) -> bool:
    text = str(value).strip()
    if text == "":
        return True
    if text.startswith("<FILL_ME_"):
        return True
    if text.startswith("<OPTIONAL_FILL_ME_"):
        return True
    return False


def _require_config_value(value: Any, field_name: str) -> str:
    if _is_placeholder(value):
        raise ValueError(
            f"Config field '{field_name}' is not filled. "
            f"Please replace the placeholder with a real path/value."
        )
    return str(value).strip()


def build_report_prompt(row_prompt: str, prompt_cfg: Dict[str, Any]) -> str:
    """
    TODO(student-1): Build the final LLM user prompt.
    Hint:
    - Combine `prompt_cfg['system']`, `prompt_cfg['user']`, and row-specific prompt text.
    - Return one string instruction used for report generation.
    """
    # START TODO(student-1)
    raise NotImplementedError(
        "TODO(student-1): implement build_report_prompt(...) in src/runners/run_eval.py"
    )
    # END TODO(student-1)


def generate_reports_batch(model: Any, image_paths: List[str], prompts: List[str]) -> List[str]:
    predictions = model.answer_vqa_batch(
        image_paths=image_paths,
        user_prompts=prompts,
    )
    return [str(p).strip() for p in predictions]

def run_eval(config: Dict[str, Any]) -> None:

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    prompt_cfg = config.get("prompt", {})

    dataset_name = dataset_cfg["name"]
    model_name   = model_cfg["name"]

    output_dir = Path(f"experiments/{dataset_name}/{model_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, output_dir)

    preprocessed_path = Path(dataset_cfg.get("preprocessed", "gpt_processed_data.jsonl"))
    if not preprocessed_path.exists():
        _require_config_value(config.get("metadata"), "metadata")
        print(f"[INFO] Preprocessed file not found, generating: {preprocessed_path}")
        chexpert_df = get_cleaned(config)
    else:
        chexpert_df = pd.read_json(preprocessed_path, lines=True)

    def normalize_prompt(value: Any) -> str:
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)

    # chexpert_df = chexpert_df[chexpert_df['split'] == 'test'].reset_index(drop=True)

    model = build_model_from_config(model_cfg)

    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        print(f"[INFO] Skipping prediction generation — file already exists: {predictions_path}")
        return
    fout = predictions_path.open("w")

    dataset_dir = Path(_require_config_value(dataset_cfg.get("base_dir"), "dataset.base_dir"))
    batch_size = config.get("batch_size", 512)
    num_rows = len(chexpert_df)
    buffer = []

    for idx, row in tqdm(
        chexpert_df.iterrows(),
        total=num_rows,
        desc="Running VQA",
        ncols=80,
    ):
        buffer.append((idx, row))

        if len(buffer) == batch_size:
            # Process and flush
            idxs, rows = zip(*buffer)
            image_paths = [str(dataset_dir / r["image"]) for r in rows]
            prompts = [
                build_report_prompt(normalize_prompt(r["prompts"]), prompt_cfg)
                for r in rows
            ]
            
            try:
                predictions = generate_reports_batch(
                    model=model,
                    image_paths=image_paths,
                    prompts=prompts,
                )
            except NotImplementedError:
                raise
            except Exception as e:
                print(f"ERROR for batch starting at idx {idxs[0]}: {e}")
                predictions = [""] * len(rows)

            for i, (row_idx, row) in enumerate(buffer):
                record = {
                    "row_id": int(row_idx),
                    "image_path": str(dataset_dir / row["image"]),
                    "impressions": str(row.get("impressions", "")),
                    "prediction": predictions[i],
                }
                fout.write(json.dumps(record) + "\n")

            buffer = []  # clear

    # Handle leftover rows in buffer
    if buffer:
        idxs, rows = zip(*buffer)
        image_paths = [str(dataset_dir / r["image"]) for r in rows]
        prompts = [
            build_report_prompt(normalize_prompt(r["prompts"]), prompt_cfg)
            for r in rows
        ]

        try:
            predictions = generate_reports_batch(
                model=model,
                image_paths=image_paths,
                prompts=prompts,
            )
        except NotImplementedError:
            raise
        except Exception as e:
            print(f"ERROR for final batch starting at idx {idxs[0]}: {e}")
            predictions = [""] * len(rows)

        for i, (row_idx, row) in enumerate(buffer):
            record = {
                "row_id": int(row_idx),
                "image_path": str(dataset_dir / row["image"]),
                "impressions": str(row.get("impressions", "")),
                "prediction": predictions[i],
            }
            fout.write(json.dumps(record) + "\n")

    fout.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_eval(cfg)


if __name__ == "__main__":
    main()
