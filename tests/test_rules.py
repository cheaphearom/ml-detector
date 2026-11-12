import numpy as np
import pandas as pd

from detector.features import SERVICES, feature_columns, feature_name
from detector.models import RuleBasedDetector, RuleConfig, sustained_min


def frame(n, step=10.0, **overrides):
    data = {c: np.zeros(n) for c in feature_columns()}
    data[feature_name("api-gateway", "http_count")] = np.full(n, 100.0)
    data.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    df = pd.DataFrame(data)
    df.insert(0, "window_start", np.arange(n) * step)
    df.insert(1, "window_end", df["window_start"] + 30)
    return df


def test_sustained_min_needs_k_consecutive_values():
    v = np.array([2, 2, 0, 2, 2, 2, 2])
    assert list(sustained_min(v, 3)) == [0, 0, 0, 0, 0, 2, 2]


def test_sustained_min_resets_on_gaps():
    v = np.array([2, 2, 2, 2])
    starts = np.array([0, 10, 40, 50])     # windows 20 and 30 are missing
    assert list(sustained_min(v, 3, starts, 10)) == [0, 0, 0, 0]


def test_rule_fires_only_after_three_consecutive_windows():
    ratio = [0.0, 0.10, 0.10, 0.10, 0.10, 0.0]
    det = RuleBasedDetector()
    df = frame(6, **{feature_name("api-gateway", "http_5xx_ratio"): ratio})
    assert list(det.predict(df)) == [0, 0, 0, 1, 1, 0]
    assert det.fired_rules(df)[3] == ["gateway_5xx_ratio"]


def test_two_windows_are_not_enough():
    det = RuleBasedDetector()
    df = frame(5, **{feature_name("api-gateway", "latency_p95"): [0, 3000, 3000, 0, 3000]})
    assert det.predict(df).sum() == 0


def test_threshold_is_strict_and_min_requests_applies():
    det = RuleBasedDetector()
    at_threshold = frame(4, **{feature_name("api-gateway", "latency_p95"): [2000] * 4})
    assert det.predict(at_threshold).sum() == 0
    few_requests = frame(4, **{feature_name("api-gateway", "http_5xx_ratio"): [1.0] * 4,
                               feature_name("api-gateway", "http_count"): [5] * 4})
    assert det.predict(few_requests).sum() == 0


def test_error_rule_per_service_and_score_is_continuous():
    det = RuleBasedDetector(RuleConfig(errors_per_service=10))
    errors = {feature_name("inventory-service", "error_count"): [11, 12, 30, 5]}
    df = frame(4, **errors)
    assert list(det.predict(df)) == [0, 0, 1, 0]
    scores = det.score(df)
    assert scores[2] == np.float64(1.1)       # min(11, 12, 30) / 10
    assert all(f"errors_{s}" in det.rule_ratios(df) for s in SERVICES)
