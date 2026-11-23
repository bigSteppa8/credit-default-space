"""
Tests run by the CI pipeline before anything is deployed.

They check that:
  1. model.json is well formed and the model meets the accuracy gate,
  2. the JavaScript used by the Space (predict.js) gives the same
     probabilities as the scikit-learn pipeline on all 6,000 test rows,
  3. the Space files that Hugging Face needs are present and consistent.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "model.json").read_text())


@pytest.fixture(scope="session")
def fitted():
    df = train.load_data()
    X_train, X_test, y_train, y_test = train.split(df)
    model = train.build_model().fit(X_train, y_train)
    return model, X_test, y_test


def test_model_json_structure(params):
    assert set(params["numeric"]) == set(train.CONTINUOUS_FEATURES)
    assert set(params["categorical"]) == set(train.ONEHOT_FEATURES)
    assert len(params["feature_order"]) == 23
    for p in params["numeric"].values():
        assert p["scale"] > 0
    assert 0 < params["threshold"] < 1


def test_accuracy_gate(params):
    assert params["test_accuracy"] >= train.MIN_TEST_ACCURACY


def test_javascript_matches_sklearn(params, fitted, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")

    model, X_test, _ = fitted
    expected = model.predict_proba(X_test)[:, 1]

    rows_file = tmp_path / "rows.json"
    rows_file.write_text(X_test.to_json(orient="records"))
    script = (
        "const m=require(process.argv[1]);"
        "const fs=require('fs');"
        "const p=JSON.parse(fs.readFileSync(process.argv[2]));"
        "const rows=JSON.parse(fs.readFileSync(process.argv[3]));"
        "process.stdout.write(JSON.stringify(rows.map(r=>m.predict(p,r).probability)));"
    )
    out = subprocess.run(
        [node, "-e", script, str(ROOT / "predict.js"), str(ROOT / "model.json"), str(rows_file)],
        capture_output=True, text=True, check=True,
    )
    got = np.array(json.loads(out.stdout))

    assert got.shape == expected.shape
    assert np.max(np.abs(got - expected)) < 1e-4
    # identical class decisions at the 0.5 threshold
    assert np.array_equal(got >= 0.5, expected >= 0.5)


def test_unknown_category_is_ignored(params):
    """OneHotEncoder(handle_unknown='ignore') -> unseen codes contribute 0."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    base = {k: 0 for k in params["feature_order"]}
    base.update({"LIMIT_BAL": 50000, "AGE": 30, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1})
    a = dict(base, EDUCATION=99)
    script = (
        "const m=require(process.argv[1]);const p=require(process.argv[2]);"
        "const r=JSON.parse(process.argv[3]);"
        "process.stdout.write(String(m.predict(p,r).contributions.EDUCATION));"
    )
    out = subprocess.run(
        [node, "-e", script, str(ROOT / "predict.js"), str(ROOT / "model.json"), json.dumps(a)],
        capture_output=True, text=True, check=True,
    )
    # unseen code adds nothing to the logit, so its contribution is minus the baseline
    assert abs(float(out.stdout) + params["categorical_baseline"]["EDUCATION"]) < 1e-12


def test_space_files_present():
    for name in ["README.md", "index.html", "predict.js", "model.json"]:
        assert (ROOT / name).is_file(), f"{name} is missing"


def test_readme_has_space_config():
    text = (ROOT / "README.md").read_text()
    header = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert header, "README.md must start with a YAML block for Hugging Face"
    assert re.search(r"^sdk:\s*static\s*$", header.group(1), re.M)


def test_index_loads_model_files():
    html = (ROOT / "index.html").read_text()
    assert 'src="predict.js"' in html
    assert 'fetch("model.json")' in html


def test_no_file_over_10mb():
    """Spaces reject plain git pushes of files over 10 MB unless LFS/Xet is used."""
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "data" in path.relative_to(ROOT).parts[:1]:
            continue
        if path.is_file():
            assert path.stat().st_size < 10 * 1024 * 1024, f"{path} is over 10 MB"
