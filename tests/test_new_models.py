"""PCA reconstruction error, Logistic Regression, method selection and the grid-edge check."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app
from detector.evaluate import evaluate_paths, make_detectors, select_methods
from detector.features import feature_columns, feature_name
from detector.models import (
    METHOD_LABELS, METHOD_ORDER, ML_METHODS, IsolationForestDetector, LogisticRegressionDetector,
    PCADetector, grid_edge_check, symptom_columns, tune_rules,
)


def windows(n, seed=0, anomalous=()):
    """Normal windows with noisy symptom features; ``anomalous`` rows get high latency and errors."""
    rng = np.random.default_rng(seed)
    data = {c: np.zeros(n) for c in feature_columns()}
    for c in symptom_columns(list(data)):
        data[c] = rng.gamma(2.0, 2.0, n)
    data[feature_name("api-gateway", "http_count")] = rng.poisson(100, n).astype(float)
    df = pd.DataFrame(data)
    df.insert(0, "window_start", np.arange(n) * 10.0)
    df.insert(1, "window_end", df["window_start"] + 30)
    y = np.zeros(n, dtype=int)
    for i in anomalous:
        df.loc[i, feature_name("api-gateway", "latency_p95")] = 5000.0
        df.loc[i, feature_name("inventory-service", "error_count")] = 80.0
        y[i] = 1
    return df, y


# --- PCA -------------------------------------------------------------------------------
def test_pca_learns_from_normal_windows_only_and_scores_anomalies_higher():
    X, y = windows(300, anomalous=range(250, 260))
    det = PCADetector().fit(X, y)
    normal_only = PCADetector().fit(X[y == 0])
    assert det.threshold_ == pytest.approx(normal_only.threshold_)
    assert 0 < det.n_components_ < len(det.columns_)
    assert det.explained_variance_ >= 0.95
    s = det.score(X)
    assert s[y == 1].min() > np.percentile(s[y == 0], 99)
    assert det.predict(X)[y == 1].all()


def test_pca_threshold_uses_the_isolation_forest_percentile_rule():
    X, y = windows(200)
    det = PCADetector(percentile=97.0).fit(X, y)
    assert det.threshold_ == pytest.approx(np.percentile(det.score(X), 97.0))
    assert abs(det.predict(X).mean() - 0.03) < 0.01
    assert make_detectors(10.0, 1)["pca"].percentile == IsolationForestDetector().percentile


def test_pca_always_keeps_a_residual():
    X, y = windows(100)
    det = PCADetector(variance=0.9999999).fit(X, y)
    assert det.n_components_ == len(det.columns_) - 1
    assert det.score(X).max() > 0


def test_pca_errors():
    X, y = windows(20, anomalous=range(12))
    with pytest.raises(ValueError, match="at least 10 normal"):
        PCADetector().fit(X, y)
    with pytest.raises(RuntimeError):
        PCADetector().score(X)
    assert PCADetector().fit(*windows(50)).score(X.iloc[:0]).size == 0


# --- Logistic Regression ---------------------------------------------------------------
def test_logistic_regression_separates_clear_faults():
    X, y = windows(300, anomalous=list(range(100, 115)) + list(range(200, 215)))
    det = LogisticRegressionDetector(random_state=1).fit(X, y)
    s = det.score(X)
    assert ((0 <= s) & (s <= 1)).all()
    assert (det.predict(X) == (s >= 0.5)).all()
    assert (det.predict(X) == y).mean() > 0.95
    coef = det.coefficients(top=5)
    assert len(coef) == 5 and abs(coef[0][1]) >= abs(coef[-1][1])
    assert det.pipeline_.named_steps["lr"].class_weight == "balanced"


def test_logistic_regression_needs_both_classes():
    X, y = windows(50)
    with pytest.raises(ValueError, match="needs labels"):
        LogisticRegressionDetector().fit(X)
    with pytest.raises(ValueError, match="both normal and anomalous"):
        LogisticRegressionDetector().fit(X, y)
    with pytest.raises(RuntimeError):
        LogisticRegressionDetector().score(X)


# --- method selection ------------------------------------------------------------------
def test_select_methods():
    assert select_methods() == list(METHOD_ORDER)
    assert select_methods(["pca"]) == ["rules", "pca"]                 # default rules always included
    assert select_methods(["pca", "rules_tuned"], tune_rules=False) == ["rules", "pca"]
    assert select_methods(["logistic_regression", "isolation_forest"]) == ["rules", "isolation_forest",
                                                                           "logistic_regression"]
    with pytest.raises(ValueError, match="unknown method"):
        select_methods(["svm"])
    assert set(METHOD_LABELS) == set(METHOD_ORDER)
    assert ML_METHODS == ("isolation_forest", "random_forest", "pca", "logistic_regression")


# --- grid-edge check -------------------------------------------------------------------
def rule_frame(n, **overrides):
    data = {c: np.zeros(n) for c in feature_columns()}
    data[feature_name("api-gateway", "http_count")] = np.full(n, 100.0)
    data.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    df = pd.DataFrame(data)
    df.insert(0, "window_start", np.arange(n) * 10.0)
    df.insert(1, "window_end", df["window_start"] + 30)
    return df


def test_grid_edge_flags_a_lower_edge_that_matters():
    p95 = np.full(40, 100.0)
    y = np.zeros(40, dtype=int)
    p95[10:13] = 1500.0
    y[10:13] = 1
    from detector.models import RULE_TUNING_GRID
    _, table = tune_rules(rule_frame(40, **{"api-gateway__latency_p95": p95}), y)
    flags = {f["parameter"]: f for f in grid_edge_check(table, RULE_TUNING_GRID)}
    assert flags["gateway_p95_ms"]["edge"] == "lower"
    assert flags["gateway_p95_ms"]["neighbour"] == 2000.0
    assert flags["gateway_p95_ms"]["neighbour_f1"] < flags["gateway_p95_ms"]["f1"]
    assert "consecutive" not in flags            # 1 is the natural minimum
    assert "gateway_5xx_ratio" not in flags      # indifferent: the neighbour scores the same


def test_grid_edge_flags_an_upper_edge():
    errors = np.full(40, 8.0)                     # normal windows: 8 ERROR lines (alerts at 3 and 5)
    y = np.zeros(40, dtype=int)
    errors[20:25] = 12.0
    y[20:25] = 1
    from detector.models import RULE_TUNING_GRID
    _, table = tune_rules(rule_frame(40, **{"inventory-service__error_count": errors}), y)
    flags = {f["parameter"]: f for f in grid_edge_check(table, RULE_TUNING_GRID)}
    assert flags["errors_per_service"]["edge"] == "upper"
    assert flags["errors_per_service"]["value"] == 10.0


def test_grid_edge_is_empty_for_an_interior_optimum():
    grid = {"consecutive": (1, 2, 3), "gateway_5xx_ratio": (0.05,), "gateway_p95_ms": (1000.0, 2000.0, 3000.0),
            "errors_per_service": (10.0,)}
    p95 = np.full(40, 1500.0)                    # normal traffic sits at 1500 ms
    y = np.zeros(40, dtype=int)
    p95[20:25] = 2500.0
    y[20:25] = 1
    cfg, table = tune_rules(rule_frame(40, **{"api-gateway__latency_p95": p95}), y, grid=grid)
    assert cfg.gateway_p95_ms == 2000.0
    assert grid_edge_check(table, grid) == []


# --- end to end ------------------------------------------------------------------------
def test_evaluation_includes_pca_and_logistic_regression(synthetic_run, tmp_path):
    out = tmp_path / "eval"
    evaluate_paths(synthetic_run / "logs", synthetic_run / "labels.csv", out_dir=out)
    saved = json.loads((out / "results.json").read_text())
    for m in ("pca", "logistic_regression"):
        assert m in saved["metrics"] and m in saved["detection_delay"]
        assert saved["metrics"][m]["roc_auc"] is not None
        assert m in saved["comparison_vs_rules"]
        assert all(m in per_ml for per_ml in saved["comparison_vs_rule_variants"].values())
        for group in saved["by_fault_type"].values():
            assert m in group
    assert saved["config"]["pca"]["n_components"] > 0
    assert saved["config"]["pca"]["percentile"] == saved["config"]["isolation_forest"]["percentile"]
    assert saved["config"]["methods"] == list(METHOD_ORDER)
    assert len(saved["logistic_regression_coefficients"]) == 20
    assert "grid_edge" in saved["rule_tuning"]
    delays = pd.read_csv(out / "detection_delays.csv")
    assert {"pca_detected", "logistic_regression_delay_s"} <= set(delays.columns)
    report = (out / "report.md").read_text()
    assert "PCA (reconstruction)" in report and "Logistic Regression coefficients" in report
    assert "Grid-edge check" in report


def test_api_evaluate_can_request_the_new_models(synthetic_run, tmp_path):
    settings = Settings(log_dir=str(synthetic_run / "logs"), labels_path=str(synthetic_run / "labels.csv"),
                        results_dir=str(tmp_path / "results"), model_dir=str(tmp_path / "models"),
                        live=False, tail_from_start=True, buffer_minutes=10)
    with TestClient(create_app(settings)) as client:
        body = client.post("/evaluate", json={"make_figures": False, "methods": ["pca", "logistic_regression"]}).json()
        assert set(body["metrics"]) == {"rules", "pca", "logistic_regression"}
        bad = client.post("/evaluate", json={"make_figures": False, "methods": ["svm"]})
        assert bad.status_code == 400
