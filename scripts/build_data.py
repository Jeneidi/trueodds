#!/usr/bin/env python3
"""Train four classifiers on UCI Adult and export calibration/test probabilities."""

from __future__ import annotations

import csv
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 2027
DATA_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "adult.json"
NUMERIC = [0, 2, 4, 10, 11, 12]
CATEGORICAL = [i for i in range(14) if i not in NUMERIC]


def load_adult() -> tuple[np.ndarray, np.ndarray]:
    """Download and parse the original UCI train and test files."""
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    rows: list[list[object]] = []
    labels: list[int] = []
    for filename in ("adult.data", "adult.test"):
        text = archive.read(filename).decode("utf-8")
        for row in csv.reader(io.StringIO(text), skipinitialspace=True):
            if len(row) != 15 or row[0].startswith("|"):
                continue
            values = [np.nan if value == "?" else value for value in row[:14]]
            for index in NUMERIC:
                values[index] = float(values[index])
            rows.append(values)
            labels.append(int(row[14].rstrip(".") == ">50K"))
    return np.asarray(rows, dtype=object), np.asarray(labels, dtype=np.int8)


def temperature_scale(prob: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(prob, 1e-12, 1 - 1e-12)
    logits = np.log(clipped / (1 - clipped)) / temperature
    return 1 / (1 + np.exp(-logits))


def fit_temperature(prob: np.ndarray, y_true: np.ndarray) -> float:
    """Minimize binary NLL over log(T) with a bounded ternary search."""
    left, right = math.log(0.05), math.log(20.0)
    for _ in range(80):
        third = (right - left) / 3
        a, b = left + third, right - third
        if log_loss(y_true, temperature_scale(prob, math.exp(a))) < log_loss(
            y_true, temperature_scale(prob, math.exp(b))
        ):
            right = b
        else:
            left = a
    return math.exp((left + right) / 2)


def pava(y: list[float]) -> list[float]:
    """Return the least-squares non-decreasing fit for equally weighted values."""
    blocks: list[list[float]] = []
    for value in y:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right_value, right_count = blocks.pop()
            left_value, left_count = blocks.pop()
            count = left_count + right_count
            blocks.append(
                [
                    (left_value * left_count + right_value * right_count) / count,
                    count,
                ]
            )
    return [value for value, count in blocks for _ in range(int(count))]


def ece(y_true: np.ndarray, prob: np.ndarray, bin_count: int = 10) -> float:
    total = 0.0
    for index in range(bin_count):
        lower, upper = index / bin_count, (index + 1) / bin_count
        mask = (prob >= lower) & ((prob < upper) if index + 1 < bin_count else (prob <= upper))
        if mask.any():
            total += mask.mean() * abs(float(prob[mask].mean() - y_true[mask].mean()))
    return total


def murphy(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float, float, float]:
    """Exact three-term decomposition, grouping identical issued probabilities."""
    mean_y = float(y_true.mean())
    reliability = resolution = 0.0
    for value in np.unique(prob):
        mask = prob == value
        weight = float(mask.mean())
        observed = float(y_true[mask].mean())
        reliability += weight * (float(value) - observed) ** 2
        resolution += weight * (observed - mean_y) ** 2
    uncertainty = mean_y * (1 - mean_y)
    return reliability, resolution, uncertainty, reliability - resolution + uncertainty


def self_check() -> None:
    """Small dependency-free-in-spirit math check, mirrored by app.js."""
    calibrated_p = np.asarray([0.25] * 4 + [0.75] * 4)
    calibrated_y = np.asarray([0, 0, 0, 1, 0, 1, 1, 1])
    assert ece(calibrated_y, calibrated_p, 4) < 1e-12

    balanced_y = np.asarray([0, 1, 0, 1])
    constant_p = np.full(4, 0.5)
    assert abs(brier_score_loss(balanced_y, constant_p) - 0.25) < 1e-12

    rel, res, unc, reconstructed = murphy(calibrated_y, calibrated_p)
    assert min(rel, res, unc) >= 0
    assert abs(brier_score_loss(calibrated_y, calibrated_p) - reconstructed) < 1e-12

    # Hand-worked PAVA: the 1,0 violation pools to 0.5,0.5.
    assert np.allclose(pava([0, 1, 0, 1]), [0, 0.5, 0.5, 1])
    assert np.all(np.diff(pava([0.8, 0.2, 0.6, 0.4])) >= 0)

    identity = temperature_scale(np.asarray([0.1, 0.5, 0.9]), 1)
    assert np.allclose(identity, [0.1, 0.5, 0.9])
    overconfident_p = np.asarray([0.02, 0.15, 0.90, 0.10, 0.85, 0.98])
    overconfident_y = np.asarray([0, 0, 0, 1, 1, 1])
    fitted_t = fit_temperature(overconfident_p, overconfident_y)
    assert log_loss(overconfident_y, temperature_scale(overconfident_p, fitted_t)) < log_loss(
        overconfident_y, overconfident_p
    )

    tie_free_p = np.asarray([0.1, 0.2, 0.8, 0.9])
    tie_free_y = np.asarray([0, 0, 1, 1])
    isotonic_p = np.asarray(pava(tie_free_y.tolist()))
    assert abs(roc_auc_score(tie_free_y, tie_free_p) - roc_auc_score(tie_free_y, isotonic_p)) < 1e-12
    print("Python self-check: 7/7 passed")


def rounded(values: np.ndarray) -> list[float]:
    return np.round(np.clip(values, 0, 1), 4).tolist()


def main() -> None:
    self_check()
    x, y = load_adult()
    all_indices = np.arange(len(y))
    fit_indices, remaining = train_test_split(
        all_indices, test_size=0.4, random_state=SEED, stratify=y
    )
    calibration_indices, test_indices = train_test_split(
        remaining, test_size=0.5, random_state=SEED, stratify=y[remaining]
    )

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUMERIC,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                CATEGORICAL,
            ),
        ]
    )
    x_fit = preprocess.fit_transform(x[fit_indices])
    x_calibration = preprocess.transform(x[calibration_indices])
    x_test = preprocess.transform(x[test_indices])

    models = {
        "logistic": (
            "Logistic regression",
            LogisticRegression(max_iter=1000, random_state=SEED),
        ),
        "forest": (
            "Random forest",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
        "boosting": (
            "Gradient boosting",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=SEED,
            ),
        ),
        # ponytail: smoothing keeps four-decimal export faithful to rank order;
        # use a larger artifact precision if sub-decimal tail ranking matters.
        "naive_bayes": ("Gaussian naive Bayes", GaussianNB(var_smoothing=0.1)),
    }

    artifact: dict[str, object] = {
        "dataset": "UCI Adult income",
        "source": "https://doi.org/10.24432/C5XW20",
        "seed": SEED,
        "fit_n": len(fit_indices),
        "calibration": {"y_true": y[calibration_indices].tolist()},
        "test": {"y_true": y[test_indices].tolist()},
        "models": {},
    }

    print(f"Dataset: UCI Adult, {len(y):,} rows")
    for key, (label, model) in models.items():
        model.fit(x_fit, y[fit_indices])
        calibration_prob = model.predict_proba(x_calibration)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
        exported_test_prob = np.round(np.clip(test_prob, 0, 1), 4)
        artifact["models"][key] = {
            "label": label,
            "calibration_prob": rounded(calibration_prob),
            "test_prob": rounded(test_prob),
        }
        print(
            f"{label:24} "
            f"accuracy={accuracy_score(y[test_indices], exported_test_prob >= 0.5):.4f} "
            f"brier={brier_score_loss(y[test_indices], exported_test_prob):.4f} "
            f"logloss={log_loss(y[test_indices], np.clip(exported_test_prob, 1e-12, 1 - 1e-12)):.4f} "
            f"auc={roc_auc_score(y[test_indices], exported_test_prob):.4f}"
        )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    assert OUT.stat().st_size < 3_000_000
    assert len(set(fit_indices) & set(calibration_indices)) == 0
    assert len(set(fit_indices) & set(test_indices)) == 0
    assert len(set(calibration_indices) & set(test_indices)) == 0
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
