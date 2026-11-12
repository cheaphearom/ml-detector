"""Rule tuning (training windows only) and the three rule variants in the evaluation."""
import json

import numpy as np
import pandas as pd
import pytest

from detector.cli import main as cli_main
from detector.evaluate import compare_with_rules, evaluate, evaluate_paths
from detector.features import feature_columns, feature_name
from detector.models import (
    LIVE_METHODS, METHOD_LABELS, METHOD_ORDER, ML_METHODS, RULE_METHODS, RULE_TUNING_GRID,
    TunedRuleBasedDetector, TunedRuleK1Detector, alert_episodes, tune_rules,
)
from detector.parsing import load_labels, load_logs

P95 = feature_name("api-gateway", "latency_p95")


def frame(n, step=10.0, **overrides):
    data = {c: np.zeros(n) for c in feature_columns()}
    data[feature_name("api-gateway", "http_count")] = np.full(n, 100.0)
    data.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    df = pd.DataFrame(data)
    df.insert(0, "window_start", np.arange(n) * step)
    df.insert(1, "window_end", df["window_start"] + 30)
    return df


def test_tuning_picks_thresholds_and_consecutive_with_best_training_f1():
    # two 3-window faults whose p95 (1500 ms) is above 1000 ms but below the 2000 ms default
    p95 = np.full(40, 100.0)
    y = np.zeros(40, dtype=int)
    for s in (10, 25):
        p95[s:s + 3] = 1500.0
        y[s:s + 3] = 1
    X = frame(40, **{P95: p95})
    cfg, table = tune_rules(X, y)
    assert cfg.gateway_p95_ms == 1000.0
    assert cfg.consecutive == 1                  # waiting for 2-3 windows only loses recall here
    assert table.iloc[0]["f1"] == 1.0
    assert len(table) == np.prod([len(v) for v in RULE_TUNING_GRID.values()])
    assert table["f1"].is_monotonic_decreasing


def test_ties_are_broken_by_fewer_false_alarm_episodes_then_grid_order():
    # nothing is detectable (every F1 is 0); single-window 2500 ms spikes in normal windows
    p95 = np.full(30, 100.0)
    p95[[5, 15, 25]] = 2500.0
    y = np.zeros(30, dtype=int)
    y[10] = 1
    cfg, table = tune_rules(frame(30, **{P95: p95}), y)
    assert (table["f1"] == 0).all()
    best = table.iloc[0]
    assert best["false_alarm_episodes"] == 0
    # first zero-false-alarm combination in grid order: consecutive=1 needs p95 3000 ms
    assert (cfg.consecutive, cfg.gateway_p95_ms) == (1, 3000.0)
    assert table["false_alarm_episodes"].max() == 3


def test_tuned_detectors_need_labels_and_k1_keeps_the_thresholds():
    p95 = np.full(40, 100.0)
    y = np.zeros(40, dtype=int)
    p95[20:26] = 2500.0
    y[20:26] = 1
    p95[[5, 12]] = 2500.0                       # two normal spikes: k=1 F1 12/14 < k=2 F1 10/11
    X = frame(40, **{P95: p95})
    with pytest.raises(ValueError, match="labels"):
        TunedRuleBasedDetector().fit(X)
    tuned = TunedRuleBasedDetector().fit(X, y)
    k1 = TunedRuleK1Detector().fit(X, y)
    assert tuned.config.consecutive == 2
    assert k1.config.consecutive == 1
    for key in ("gateway_5xx_ratio", "gateway_p95_ms", "errors_per_service"):
        assert getattr(k1.config, key) == getattr(tuned.config, key)
    assert tuned.tuning_["selected"]["consecutive"] == 2
    assert tuned.tuning_["train_windows"] == 40
    assert k1.predict(X)[5] == 1 and tuned.predict(X)[5] == 0
    assert tuned.tuning_["train_f1"] == pytest.approx(10 / 11)


def test_method_lists():
    assert METHOD_ORDER == RULE_METHODS + ML_METHODS
    assert RULE_METHODS == ("rules", "rules_tuned", "rules_tuned_k1")
    assert set(METHOD_LABELS) == set(METHOD_ORDER)
    assert "rules_tuned" not in LIVE_METHODS     # the live detector keeps its three methods


def test_alert_episodes_is_reexported_from_evaluate():
    from detector import evaluate as ev
    assert ev.alert_episodes is alert_episodes


def test_compare_with_a_chosen_rule_variant():
    delays = pd.DataFrame({
        "rules_detected": [False, False], "rules_delay_s": [np.nan, np.nan],
        "rules_tuned_detected": [True, True], "rules_tuned_delay_s": [10.0, 50.0],
        "random_forest_detected": [True, True], "random_forest_delay_s": [20.0, 30.0],
    })
    vs_default = compare_with_rules(delays, "random_forest")
    vs_tuned = compare_with_rules(delays, "random_forest", rule="rules_tuned")
    assert vs_default["earlier"] == 2 and vs_default["detected_only_by_ml"] == 2
    assert (vs_tuned["earlier"], vs_tuned["later_or_missed"]) == (1, 1)
    assert vs_tuned["mean_time_saved_s"] == pytest.approx((-10 + 20) / 2)


def test_evaluation_reports_three_rule_variants(synthetic_run, tmp_path):
    out = tmp_path / "eval"
    results = evaluate_paths(synthetic_run / "logs", synthetic_run / "labels.csv", out_dir=out)
    saved = json.loads((out / "results.json").read_text())
    for m in RULE_METHODS:
        assert m in saved["metrics"] and m in saved["detection_delay"]
        assert m in saved["config"]
    assert set(saved["comparison_vs_rule_variants"]) == set(RULE_METHODS)
    for per_ml in saved["comparison_vs_rule_variants"].values():
        assert set(per_ml) == set(ML_METHODS)
    tuning = saved["rule_tuning"]
    assert tuning["train_windows"] == saved["dataset"]["windows_train"]
    assert tuning["selected"] == {k: saved["config"]["rules_tuned"][k] for k in RULE_TUNING_GRID}
    assert saved["config"]["rules_tuned_k1"]["consecutive"] == 1
    report = (out / "report.md").read_text()
    assert "## Rule variants" in report and "ML compared with each rule variant" in report
    assert "training windows only" in report
    delays = pd.read_csv(out / "detection_delays.csv")
    assert {"rules_tuned_detected", "rules_tuned_k1_delay_s"} <= set(delays.columns)


def test_tuning_never_sees_test_labels(synthetic_run):
    """Dropping every test-period fault from the labels must not change the tuned rules."""
    logs = load_logs(synthetic_run / "logs")
    labels = load_labels(synthetic_run / "labels.csv")
    full = evaluate(logs, labels, make_figures=False)
    split = pd.Timestamp(full["dataset"]["split_at"])
    train_only = labels[labels["start"] < split]          # keep every fault that touches training
    assert len(train_only) < len(labels)
    reduced = evaluate(logs, train_only, make_figures=False)
    assert reduced["rule_tuning"]["selected"] == full["rule_tuning"]["selected"]
    assert reduced["rule_tuning"]["train_f1"] == full["rule_tuning"]["train_f1"]


def test_custom_grid_and_disabling_tuning(synthetic_run):
    logs = load_logs(synthetic_run / "logs")
    labels = load_labels(synthetic_run / "labels.csv")
    grid = {"consecutive": (1,), "gateway_5xx_ratio": (0.05,), "gateway_p95_ms": (2000.0,),
            "errors_per_service": (3.0, 10.0)}
    res = evaluate(logs, labels, make_figures=False, rule_grid=grid)
    assert res["rule_tuning"]["candidates"] == 2
    assert res["config"]["rules_tuned"]["consecutive"] == 1
    plain = evaluate(logs, labels, make_figures=False, tune_rules=False)
    assert set(plain["metrics"]) == {"rules", *ML_METHODS}
    assert plain["rule_tuning"] is None
    assert set(plain["comparison_vs_rule_variants"]) == {"rules"}


def test_cli_flag_skips_rule_tuning(synthetic_run, tmp_path, capsys):
    out = tmp_path / "cli"
    code = cli_main(["evaluate", "--logs", str(synthetic_run / "logs"), "--labels",
                     str(synthetic_run / "labels.csv"), "--out", str(out), "--no-figures", "--no-rule-tuning"])
    assert code == 0
    assert set(json.loads((out / "results.json").read_text())["metrics"]) == {"rules", *ML_METHODS}
    code = cli_main(["evaluate", "--logs", str(synthetic_run / "logs"), "--labels",
                     str(synthetic_run / "labels.csv"), "--out", str(out), "--no-figures"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "vs rules_tuned_k1" in printed and "tuned rules (training windows only)" in printed
