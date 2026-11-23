"""
Optional Gradio version of the Space.

Hugging Face now requires a PRO (paid) account to create Gradio Spaces on
CPU hardware, so the main deployment is the free Static Space in the repo
root. If you do have PRO: create a Gradio Space (hardware: CPU Basic), and
upload this app.py, this folder's requirements.txt and the repo's model.json.

It uses the same model.json as the static site, so both versions give
identical predictions.
"""

import json
import math
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.json" if (HERE / "model.json").exists() else HERE.parent / "model.json"
PARAMS = json.loads(MODEL_PATH.read_text())

MONTHS = ["September", "August", "July", "June", "May", "April"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
STATUS = [
    ("No consumption", -2), ("Paid in full", -1), ("Revolving credit", 0),
    ("1 month late", 1), ("2 months late", 2), ("3 months late", 3), ("4 months late", 4),
    ("5 months late", 5), ("6 months late", 6), ("7 months late", 7), ("8+ months late", 8),
]


def probability(record: dict) -> float:
    """StandardScaler + OneHotEncoder(ignore unknown) + LogisticRegression."""
    z = PARAMS["intercept"]
    for name, p in PARAMS["numeric"].items():
        z += (float(record[name]) - p["mean"]) / p["scale"] * p["coef"]
    for name, weights in PARAMS["categorical"].items():
        z += weights.get(str(int(record[name])), 0.0)
    return 1 / (1 + math.exp(-z))


def predict(limit_bal, age, sex, education, marriage, *rest):
    status, bills, paid = rest[:6], rest[6:12], rest[12:18]
    row = {"LIMIT_BAL": limit_bal, "AGE": age, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage}
    row.update(dict(zip(PAY_COLS, status)))
    row.update({f"BILL_AMT{i + 1}": v for i, v in enumerate(bills)})
    row.update({f"PAY_AMT{i + 1}": v for i, v in enumerate(paid)})
    p = probability(row)
    return {"Default next month": p, "No default": 1 - p}


inputs = [
    gr.Number(label="Credit limit (NT$)", value=50000),
    gr.Number(label="Age", value=30),
    gr.Dropdown([("Male", 1), ("Female", 2)], label="Sex", value=2),
    gr.Dropdown([("Graduate school", 1), ("University", 2), ("High school", 3), ("Other", 4)],
                label="Education", value=2),
    gr.Dropdown([("Married", 1), ("Single", 2), ("Other", 3)], label="Marital status", value=2),
]
inputs += [gr.Dropdown(STATUS, label=f"Repayment status, {m}", value=0) for m in MONTHS]
inputs += [gr.Number(label=f"Bill amount, {m}", value=20000) for m in MONTHS]
inputs += [gr.Number(label=f"Amount paid, {m}", value=2000) for m in MONTHS]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Label(label="Prediction"),
    title="Credit Card Default Predictor",
    description="Logistic regression trained on the UCI Default of Credit Card Clients dataset. Coursework demo only.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
