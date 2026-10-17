"""Human-labelled FOMC sentences for validating classifiers.

Trillion Dollar Words (Shah, Paturi and Chava, ACL 2023) provides FOMC
sentences labelled hawkish, dovish or neutral by annotators. The dataset is
licensed CC BY-NC 4.0, so it is downloaded at run time rather than
redistributed here.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

REPO = "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/training_data/test-and-training"
FILES = {
    "train": REPO + "/training_data/lab-manual-combine-train-5768.xlsx",
    "test": REPO + "/test_data/lab-manual-combine-test-5768.xlsx",
}
LABELS = {0: "dovish", 1: "hawkish", 2: "neutral"}
ORDER = ["dovish", "neutral", "hawkish"]


def load_split(split: str, data_dir: str | Path = "data/tdw") -> pd.DataFrame:
    path = Path(data_dir) / Path(FILES[split]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FILES[split], path)
    df = pd.read_excel(path)
    return pd.DataFrame({"sentence": df["sentence"].astype(str), "label": df["label"].map(LABELS), "year": df["year"]})


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=ORDER)),
        "confusion": confusion_matrix(y_true, y_pred, labels=ORDER).tolist(),
        "n": int(len(y_true)),
    }
