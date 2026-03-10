import json
from pathlib import Path

import pandas as pd
from cleantext import clean

from src.data.preprocessing import CheXpertPreprocessing

#The clean-text 0.6.0 package is used to clean the text fields. By default, it transliterates non-ASCII characters to 
# closest ASCII representation, fixes unicode sequences, lowercases the text, and removes line breaks.

#Utility Methods
def create_labels_dict(row):
    """Convert CheXpert labels to dictionary for a single row"""
    return {label: row[label] for label in CHEXPERT_LABELS if label in row}

def clean_text(value):
    text = str(value)
    try:
        return clean(text, no_line_breaks=True, lower=False)
    except Exception:
        # Fallback for cleantext variants that expose a different signature
        # or require unavailable NLTK resources.
        return text.replace("\n", " ").replace("\r", " ").strip()

def create_gt_list(findings, impression):
    gt = []
    output = ""
    if findings and findings != 'nan' and findings.lower() != 'none':
        output = output + "Findings: " + str(findings).replace("\n", "").strip() + " "
    if impression and impression != 'nan' and impression.lower() != 'none':
        output = output + "Impression: " + str(impression).replace("\n", "").strip()
    dict_findings = {"from": "gpt", "value": output}
    gt.append(dict_findings)
    return gt

def create_prompt_list(indication, examination):
    conversations = []
    if indication is not None and examination is not None:
        reason = indication.replace('\n', '')
        dict_reason = f"Provide a description of the findings and impression in the radiology image given the following indication: {reason}. The examination conducted was {examination}"
    elif indication is None and examination is not None:
        dict_reason = f"Provide a description of the findings and impression in the radiology image. The examination conducted was {examination}"
    elif indication is not None and examination is None:
        reason = indication.replace('\n', '')
        dict_reason = f"Provide a description of the findings and impression in the radiology image given the following indication: {reason}"
    else:
        dict_reason = "Provide a description of the findings and impression in the radiology image."
    conversations.append(dict_reason)
    return conversations

#Set paths and constants
CHEXPERT_LABELS = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
    'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
    'Pleural Effusion','Pleural Other','Fracture','Support Devices']

def get_cleaned(config):
    #Load and process CheXpert metadata to fit the model input format.
    path = config.get("metadata", {})
    splits_path = config.get("splits", {})
    dataset_cfg = config.get("dataset", {})
    preprocessed_path = Path(dataset_cfg.get("preprocessed", "gpt_processed_data.jsonl"))
    generate_method = config.get("generate_method", "gpt")

    chexpert_obj = CheXpertPreprocessing(path, splits_path)
    df = chexpert_obj.read_data()
    df['section_indication'] = (df['section_history'].fillna('') + " " + df['section_clinical_history'].fillna(''))
    df['chexpert_labels'] = df.apply(create_labels_dict, axis=1)

    # Printing out some useful metrics
    print("Number of null findings: ", df['section_findings'].isnull().sum())
    print("Number of null impressions: ", df['section_impression'].isnull().sum())
    print("Length of chexpert data after preprocessing: ", len(df))
    print(df['split'].value_counts())

    def chexpert_record(index):
        """Process a single CheXpert row into the preprocessed schema."""
        row = df.iloc[index]
        reason = clean_text(row['section_indication'])
        findings = clean_text(row['section_findings'])
        impressions = clean_text(row['section_impression'])
        examination = clean_text(row['section_narrative'])
        input_row = {
            'reason': reason,
            'findings': findings,
            'impressions': impressions,
            'examination': examination,
            'image': row['path_to_image'],
            'generate_method': generate_method,
            'chexpert_labels': row['chexpert_labels'],
            'split': row['split'],
            'prompts': create_prompt_list(reason, examination),
            'gt': create_gt_list(findings, impressions)
        }
        return input_row

    records = [chexpert_record(i) for i in range(len(df))]

    input_json_df = pd.DataFrame(records)
    assert len(df) == len(input_json_df), "Mismatch in number of records processed"
    print("Generate_Method value counts in processed GPT data: ", input_json_df['generate_method'].value_counts())

    def write_records(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    try:
        write_records(preprocessed_path)
    except OSError as e:
        fallback_path = Path("gpt_processed_data.jsonl")
        if fallback_path == preprocessed_path:
            raise
        print(f"[WARN] Could not write to {preprocessed_path}: {e}")
        print(f"[INFO] Falling back to local file: {fallback_path}")
        write_records(fallback_path)
        preprocessed_path = fallback_path

    print(f"[INFO] Wrote preprocessed records to: {preprocessed_path}")

    return input_json_df
