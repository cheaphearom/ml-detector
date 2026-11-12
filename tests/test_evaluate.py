import json

import numpy as np
import pandas as pd

from detector.evaluate import alert_episodes, compare_with_rules, evaluate_paths, prepare_dataset
from detector.models import METHOD_ORDER, ML_METHODS
from detector.parsing import load_labels, load_logs


def test_full_evaluation_on_synthetic_logs(synthetic_run, tmp_path):
    out = tmp_path / "eval"
    results = evaluate_paths(synthetic_run / "logs", synthetic_run / "labels.csv", out_dir=out)

    for name in ("results.json", "report.md", "detection_delays.csv", "timeline.png", "metrics.png",
                 "detection_delay.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name

    saved = json.loads((out / "results.json").read_text())
    assert set(saved["metrics"]) == set(METHOD_ORDER)
    for m in METHOD_ORDER:
        r = saved["metrics"][m]
        for k in ("precision", "recall", "f1", "false_positive_rate"):
            assert 0.0 <= r[k] <= 1.0
        assert r["tp"] + r["fp"] + r["tn"] + r["fn"] == saved["dataset"]["windows_test"]
    # the supervised model must learn something on these clearly visible faults
    assert saved["metrics"]["random_forest"]["f1"] > 0.5
    assert saved["dataset"]["faults_test"] >= 1
    assert len(saved["feature_importance"]) == 20

    delays = pd.read_csv(out / "detection_delays.csv")
    assert len(delays) == saved["dataset"]["faults_test"]
    for m in METHOD_ORDER:
        detected = delays.loc[delays[f"{m}_detected"], f"{m}_delay_s"]
        assert (detected > 0).all()

    report = (out / "report.md").read_text()
    assert "Q1 / Q2" in report and "Q3" in report
    assert set(results["comparison_vs_rules"]) == set(ML_METHODS)


def test_split_is_chronological_without_overlap(synthetic_run):
    logs = load_logs(synthetic_run / "logs")
    labels = load_labels(synthetic_run / "labels.csv")
    ds = prepare_dataset(logs, labels, train_frac=0.6)
    assert ds.train["window_end"].max() <= ds.t_split <= ds.test["window_start"].min()
    assert len(ds.train) + len(ds.test) < len(ds.windows)   # windows crossing the split were dropped
    assert ds.train["label"].sum() > 0


def test_alert_episodes_merge_consecutive_windows():
    pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    starts = np.array([0, 10, 20, 30, 40, 50, 70, 80], dtype=float)
    assert alert_episodes(pred, starts, 10) == [(1, 2), (4, 4), (6, 7)]


def test_compare_with_rules_buckets():
    delays = pd.DataFrame({
        "rules_detected": [True, True, True, False, False],
        "rules_delay_s": [60.0, 30.0, 20.0, np.nan, np.nan],
        "random_forest_detected": [True, True, True, True, False],
        "random_forest_delay_s": [20.0, 30.0, 40.0, 10.0, np.nan],
    })
    c = compare_with_rules(delays, "random_forest")
    assert (c["earlier"], c["same"], c["later_or_missed"], c["both_missed"]) == (2, 1, 1, 1)
    assert c["faults_detected_by_both"] == 3
    assert c["mean_time_saved_s"] == (40 + 0 - 20) / 3
    assert c["detected_only_by_ml"] == 1


def _copy_run(src, dst, meta=None):
    import shutil
    shutil.copytree(src / "logs", dst / "logs")
    shutil.copy(src / "labels.csv", dst / "labels.csv")
    if meta is not None:
        (dst / "experiment.json").write_text(json.dumps(meta))


def test_refuses_an_experiment_that_is_still_running(synthetic_run, tmp_path):
    import pytest
    _copy_run(synthetic_run, tmp_path)                      # no experiment.json yet
    with pytest.raises(ValueError, match="still running"):
        evaluate_paths(tmp_path / "logs", tmp_path / "labels.csv", make_figures=False)
    results = evaluate_paths(tmp_path / "logs", tmp_path / "labels.csv", make_figures=False,
                             allow_incomplete=True)
    assert results["metrics"]


def test_refuses_an_interrupted_experiment(synthetic_run, tmp_path):
    import pytest
    meta = json.loads((synthetic_run / "experiment.json").read_text())
    _copy_run(synthetic_run, tmp_path, {**meta, "completed": False})
    with pytest.raises(ValueError, match="stopped before it finished"):
        evaluate_paths(tmp_path / "logs", tmp_path / "labels.csv", make_figures=False)


def test_missing_labels_file_gives_a_clear_error(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="labels file not found"):
        evaluate_paths(tmp_path, tmp_path / "labels.csv")
