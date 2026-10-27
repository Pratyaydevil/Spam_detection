import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .email_classifier import MODEL_PATH, build_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "email-classification"
ENRON_URL = (
    "https://huggingface.co/datasets/adamlouly/enron_spam_data/"
    "resolve/099c1b164c1ef9ff0e7986bfb8f1b33d3ff8596a/train.csv"
)
PHISHING_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/"
    "resolve/34085a032c123ca237f314a01a67909cdea35e34/Phishing_Email.csv"
)


def download_dataset(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        request = urllib.request.Request(url, headers={"User-Agent": "LocalEmailDefender/1.0"})
        temporary_path = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary_path.replace(destination)
    return destination


def load_labeled_emails(enron_path: Path, phishing_path: Path) -> pd.DataFrame:
    enron = pd.read_csv(enron_path)
    if not {"email", "label"}.issubset(enron.columns):
        raise ValueError("Enron dataset must contain 'email' and 'label' columns.")
    enron_labels = pd.to_numeric(enron["label"], errors="coerce")
    distinct_enron_labels = set(enron_labels.dropna().unique())
    if (
        enron_labels.isna().any()
        or not distinct_enron_labels.issubset({0, 1})
        or not distinct_enron_labels
    ):
        raise ValueError(f"Unexpected Enron labels: {sorted(distinct_enron_labels)}")
    enron_rows = pd.DataFrame({
        "text": enron["email"].fillna("").astype(str),
        "label": enron_labels.map({0: "Ham", 1: "Spam"}),
    })

    phishing = pd.read_csv(phishing_path)
    if not {"Email Text", "Email Type"}.issubset(phishing.columns):
        raise ValueError("Phishing dataset must contain 'Email Text' and 'Email Type' columns.")
    phishing_types = phishing["Email Type"].fillna("").astype(str).str.strip().str.lower()

    def map_phishing_label(label: str) -> str:
        if "phishing" in label:
            return "Phishing"
        if "safe" in label or "legitimate" in label or "ham" in label:
            return "Ham"
        raise ValueError(f"Unexpected Phishing Email dataset label: {label!r}")

    phishing_rows = pd.DataFrame({
        "text": phishing["Email Text"].fillna("").astype(str),
        "label": phishing_types.map(map_phishing_label),
    })
    combined = pd.concat([enron_rows, phishing_rows], ignore_index=True)
    combined["text"] = combined["text"].str.strip()
    combined = combined.dropna(subset=["label"])
    combined = combined[combined["text"].str.len() >= 20]
    combined["_normalized"] = combined["text"].map(normalize_email)

    # Drop cross-source duplicates to keep copies of an email out of both splits.
    combined = combined.drop_duplicates("_normalized", keep="first").drop(columns="_normalized")
    present = set(combined["label"].dropna().unique())
    expected = {"Ham", "Spam", "Phishing"}
    if present != expected:
        raise ValueError(f"Expected all three classes {sorted(expected)}, found {sorted(present)}")
    return combined.reset_index(drop=True)


def normalize_email(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def split_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_and_validation, test = train_test_split(
        data,
        test_size=0.15,
        random_state=42,
        stratify=data["label"],
    )
    train, validation = train_test_split(
        train_and_validation,
        test_size=(0.15 / 0.85),
        random_state=42,
        stratify=train_and_validation["label"],
    )
    return train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local three-class email text classifier.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap on rows per class for a quick run; 0 uses all available rows.",
    )
    args = parser.parse_args()
    if args.max_per_class < 0:
        parser.error("--max-per-class must be zero or greater.")

    enron_path = download_dataset(ENRON_URL, RAW_DATA_DIR / "enron_train.csv")
    phishing_path = download_dataset(PHISHING_URL, RAW_DATA_DIR / "phishing_email.csv")
    data = load_labeled_emails(enron_path, phishing_path)
    if args.max_per_class:
        data = (
            data.groupby("label", group_keys=False)
            .apply(lambda rows: rows.sample(min(len(rows), args.max_per_class), random_state=42))
            .reset_index(drop=True)
        )

    train, validation, test = split_dataset(data)
    classifier = build_classifier()
    classifier.fit(train["text"], train["label"])

    validation_predictions = classifier.predict(validation["text"])
    test_predictions = classifier.predict(test["text"])
    report = {
        "dataset_sources": {
            "enron_spam": {
                "url": ENRON_URL,
                "license": "Apache-2.0",
                "mapping": {"0": "Ham", "1": "Spam"},
            },
            "phishing_email": {
                "url": PHISHING_URL,
                "license": "LGPL-3.0",
                "mapping": {"Safe Email": "Ham", "Phishing Email": "Phishing"},
            },
        },
        "sample_count_after_deduplication": len(data),
        "class_counts": data["label"].value_counts().sort_index().to_dict(),
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "validation": {
            "accuracy": accuracy_score(validation["label"], validation_predictions),
            "macro_f1": f1_score(validation["label"], validation_predictions, average="macro"),
        },
        "test": {
            "accuracy": accuracy_score(test["label"], test_predictions),
            "macro_f1": f1_score(test["label"], test_predictions, average="macro"),
            "classification_report": classification_report(
                test["label"], test_predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix_labels": list(classifier.named_steps["classifier"].classes_),
            "confusion_matrix": confusion_matrix(
                test["label"],
                test_predictions,
                labels=classifier.named_steps["classifier"].classes_,
            ).tolist(),
        },
        "model": "TF-IDF word and character n-grams with SGDClassifier",
        "warning": "Metrics are for these source datasets only; they do not guarantee performance on current real-world email.",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    metrics_path = MODEL_PATH.with_name("training_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Trained model saved to: {MODEL_PATH}")
    print(f"Evaluation report saved to: {metrics_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
