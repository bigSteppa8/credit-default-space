---
title: Credit Card Default Predictor
emoji: 💳
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Logistic regression predicting credit card default
---

# Credit Card Default Predictor

A logistic regression model (AAI6610, Module 8) that estimates the probability a
credit card client defaults on next month's payment. It was trained on the UCI
*Default of Credit Card Clients* dataset (Yeh & Lien, 2009): 30,000 clients,
23 features, 80/20 stratified split, `random_state=20`.

| Metric (6,000 test clients) | Value |
|---|---|
| Accuracy | 0.820 |
| Precision (default) | 0.68 |
| Recall (default) | 0.35 |

## How it is deployed

This is a **Static Space**: `index.html` loads `model.json` (the fitted scaler,
encoder and regression parameters exported by `train.py`) and `predict.js`
computes the prediction in the visitor's browser. No server is needed, so it runs
free on any Hugging Face account.

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push and pull request:

1. **test**: installs dependencies, re-trains the model with `train.py --check`
   (fails if accuracy drops below 0.80 or if the committed `model.json` no longer
   matches a fresh training run), then runs `pytest`, which checks that
   `predict.js` matches scikit-learn on all 6,000 test rows.
2. **deploy** (only on `main`, only if tests pass): pushes the repository to
   the Hugging Face Space using the `HF_TOKEN` secret.

## Run locally

```bash
pip install -r requirements-dev.txt
python train.py          # re-train and export model.json
pytest -q                # run the tests
python -m http.server    # then open http://localhost:8000
```

*Coursework demo only. Not for real lending decisions.*
