"""
Train the Module 8 logistic regression model and export it for deployment.

This reproduces the logistic regression pipeline from the AAI6610 notebook
exactly (same features, same preprocessing, same split, same random seed) and
exports the fitted parameters to model.json, which the Space reads.

Plain JSON is used instead of a .joblib pickle because (a) the static Space
runs in the browser, which cannot unpickle scikit-learn objects, (b) Hugging
Face rejects binary files pushed without Git LFS/Xet, and (c) a pickle only
loads reliably under the same scikit-learn version it was saved with.

Usage:
    python train.py                 # train, evaluate, write model.json
    python train.py --check         # train and compare with the committed
                                    # model.json instead of overwriting it (CI)
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 20
MIN_TEST_ACCURACY = 0.80  # quality gate used by the CI pipeline

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "mtdonovan207/AAI6610/"
    "012bfde756b609190bea4ac2cd438e7a54b9cc69/"
    "default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
)
TARGET = "default payment next month"

CONTINUOUS_FEATURES = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
PAYMENT_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
ONEHOT_FEATURES = CATEGORICAL_FEATURES + PAYMENT_STATUS_FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "model.json"
CACHE = ROOT / "data" / "credit.zip"


def load_data() -> pd.DataFrame:
    """Download (or reuse a cached copy of) the UCI credit card default data."""
    if CACHE.exists():
        content = CACHE.read_bytes()
    else:
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        content = response.content
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(content)

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        excel_file = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))][0]
        with z.open(excel_file) as f:
            return pd.read_excel(f, header=1)


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ONEHOT_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def split(df: pd.DataFrame):
    X = df.drop(columns=["ID", TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def export_parameters(model: Pipeline, accuracy: float, X_train: pd.DataFrame) -> dict:
    """Pull every fitted number out of the pipeline into plain JSON."""
    pre = model.named_steps["preprocessor"]
    scaler = pre.named_transformers_["num"]
    encoder = pre.named_transformers_["cat"]
    clf = model.named_steps["classifier"]

    coef = clf.coef_[0].tolist()
    n_num = len(CONTINUOUS_FEATURES)

    numeric = {
        name: {
            "mean": float(scaler.mean_[i]),
            "scale": float(scaler.scale_[i]),
            "coef": coef[i],
        }
        for i, name in enumerate(CONTINUOUS_FEATURES)
    }

    categorical = {}
    # baseline = average weight of a training client, so the "drivers" shown in
    # the app compare a category with a typical client rather than with zero
    categorical_baseline = {}
    pos = n_num
    for name, cats in zip(ONEHOT_FEATURES, encoder.categories_):
        weights = {}
        for c in cats:
            weights[str(int(c))] = coef[pos]
            pos += 1
        categorical[name] = weights
        freq = X_train[name].value_counts(normalize=True)
        categorical_baseline[name] = float(sum(freq.get(c, 0) * weights[str(int(c))] for c in cats))
    assert pos == len(coef), "coefficient count does not match feature layout"

    return {
        "model": "LogisticRegression (scikit-learn Pipeline: StandardScaler + OneHotEncoder)",
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "test_accuracy": round(float(accuracy), 6),
        "intercept": float(clf.intercept_[0]),
        "threshold": 0.5,
        "feature_order": CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + PAYMENT_STATUS_FEATURES,
        "numeric": numeric,
        "categorical": categorical,
        "categorical_baseline": categorical_baseline,
    }


def max_param_difference(a: dict, b: dict) -> float:
    diffs = [abs(a["intercept"] - b["intercept"])]
    for k, v in a["numeric"].items():
        w = b["numeric"][k]
        diffs += [abs(v["mean"] - w["mean"]), abs(v["scale"] - w["scale"]), abs(v["coef"] - w["coef"])]
    for k, cats in a["categorical"].items():
        if set(cats) != set(b["categorical"][k]):
            return float("inf")
        diffs += [abs(cats[c] - b["categorical"][k][c]) for c in cats]
    return max(diffs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify committed model.json instead of overwriting it")
    args = parser.parse_args()

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"scikit-learn {sklearn.__version__}")
    print(f"Training rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Logistic Regression test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    if accuracy < MIN_TEST_ACCURACY:
        print(f"FAIL: accuracy {accuracy:.4f} is below the {MIN_TEST_ACCURACY} quality gate")
        return 1

    params = export_parameters(model, accuracy, X_train)

    if args.check:
        committed = json.loads(MODEL_JSON.read_text())
        diff = max_param_difference(params, committed)
        print(f"Largest difference vs committed model.json: {diff:.2e}")
        if diff > 1e-4:
            print("FAIL: committed model.json does not match a fresh training run. "
                  "Run `python train.py` and commit the new model.json.")
            return 1
        print("OK: committed model.json matches a fresh training run")
        return 0

    MODEL_JSON.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Wrote {MODEL_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
