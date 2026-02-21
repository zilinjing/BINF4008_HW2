## Homework: LLM for Chest X-ray Report Generation

This repository is a student version of a CheXpert evaluation pipeline.
The goal is to implement core LLM steps for:
1. report generation from CXR images,
2. label extraction from generated reports,
3. report/label evaluation.

## Pipeline
1. `src/runners/run_eval.py`: generate radiology reports into `predictions.jsonl`.
2. `src/runners/extract_chexpert_labels.py`: extract CheXpert labels from generated reports into `labels.csv`.
3. `src/runners/compute_metrics.py`: compute label metrics and BERTScore metrics.

## Student Fill-In Map

Config placeholders:
- `configs/chexpert.yaml`: fill 2 required paths.
  - `dataset.base_dir`
  - `metadata`
- `configs/compute_metrics.yaml`: fill 3 required paths (+1 optional).
  - `metrics.labels_path`
  - `metrics.predictions_path`
  - `metrics.output_dir`

Code TODO blocks:
- `src/runners/run_eval.py`: 1 TODO
  - `TODO(student-1)`: build final report-generation prompt (prompt formatting).
- `src/runners/extract_chexpert_labels.py`: 2 TODOs
  - `TODO(student-extract-1)`: implement `load_transformers_model(...)`.
  - `TODO(student-extract-2)`: implement `batch_extract(...)`.
- `src/runners/compute_metrics.py`: 2 TODOs
  - `TODO(student-metrics-1)`: implement `_compute_prf(...)`.
  - `TODO(student-metrics-2)`: finish macro/micro aggregation in `_compute_label_metrics(...)`.


## Run Templates

1) Report generation:
```bash
python -m src.runners.run_eval --config configs/chexpert.yaml
```

2) CheXpert label extraction:
```bash
python -m src.runners.extract_chexpert_labels \
  --predictions <FILL_ME_PREDICTIONS_JSONL> \
  --ground-truth <FILL_ME_CHEXPERT_VALID_CSV> \
  --output-csv <FILL_ME_OUTPUT_LABELS_CSV> \
  --include-text
```

3) Metric computation:
```bash
python -m src.runners.compute_metrics --config configs/compute_metrics.yaml
```

## Notes
- Scripts will raise clear errors if config placeholders are not filled.
- Focus your implementation on the TODO blocks listed above.
