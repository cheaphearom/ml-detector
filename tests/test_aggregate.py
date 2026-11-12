"""Aggregation of several evaluated runs (``detector.cli aggregate``)."""
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from detector.aggregate import (
    Run, aggregate, check_same_settings, expand_runs, load_run, load_runs, pooled_comparisons, pooled_delays, pooled_wide,
    severity_of, summary_metrics,
)
from detector.cli import main as cli_main
from detector.evaluate import compare_with_rules, evaluate_paths
from detector.models import METHOD_ORDER
from detector.synthetic import generate


# --- fabricated runs (exact numbers) ---------------------------------------------------
def metrics(f1, auc=0.9, fah=1.0):
    return {"precision": f1, "recall": f1, "f1": f1, "roc_auc": auc, "false_positive_rate": 0.01,
            "false_alarms_per_hour": fah}


def fake_run(run_id, method_metrics, delays):
    return Run(run_id=run_id, path=Path(run_id), results={"metrics": method_metrics}, delays=pd.DataFrame(delays))


def delays_frame(n, methods, seed=0, params='{"severity": "low"}'):
    rng = np.random.default_rng(seed)
    d = {"fault_id": [f"F{i:03d}" for i in range(n)], "service": ["order-service"] * n,
         "type": (["LATENCY", "ERROR"] * n)[:n], "start": ["2026-01-01T00:00:00+00:00"] * n,
         "end": ["2026-01-01T00:01:00+00:00"] * n, "duration_s": [60.0] * n, "params": [params] * n}
    for m in methods:
        det = rng.random(n) < 0.8
        d[f"{m}_detected"] = det
        d[f"{m}_delay_s"] = np.where(det, rng.integers(5, 60, n).astype(float), np.nan)
    return d


def test_severity_of():
    assert severity_of('{"severity": "high", "latencyMs": 3000}') == "high"
    assert severity_of("") == "unknown"
    assert severity_of(np.nan) == "unknown"
    assert severity_of("not json") == "unknown"
    assert severity_of('{"latencyMs": 1}') == "unknown"


def test_summary_metrics_mean_and_sample_sd():
    runs = [fake_run("run_1", {"rules": metrics(0.5), "random_forest": metrics(0.8, auc=None)}, {}),
            fake_run("run_2", {"rules": metrics(0.7), "random_forest": metrics(0.9, auc=0.95)}, {}),
            fake_run("run_3", {"rules": metrics(0.9)}, {})]
    s = summary_metrics(runs).set_index("method")
    assert list(s.index) == ["rules", "random_forest"]           # METHOD_ORDER
    assert s.at["rules", "runs"] == 3
    assert s.at["rules", "f1_mean"] == pytest.approx(0.7)
    assert s.at["rules", "f1_std"] == pytest.approx(np.std([0.5, 0.7, 0.9], ddof=1))
    assert s.at["random_forest", "runs"] == 2
    assert s.at["random_forest", "roc_auc_mean"] == pytest.approx(0.95)   # missing AUC skipped
    assert np.isnan(s.at["random_forest", "roc_auc_std"])                 # one value: no SD


def test_pooled_delays_long_format_and_missing_methods():
    r1 = fake_run("run_1", {"rules": metrics(0.5), "pca": metrics(0.6)},
                  delays_frame(4, ["rules", "pca"], params='{"severity": "medium"}'))
    r2 = fake_run("run_2", {"rules": metrics(0.5)}, delays_frame(3, ["rules"], seed=1))
    long = pooled_delays([r1, r2])
    assert len(long) == 4 * 2 + 3 * 1                       # pca was not evaluated in run_2
    assert set(long.loc[long["run_id"] == "run_2", "method"]) == {"rules"}
    assert set(long.loc[long["run_id"] == "run_1", "severity"]) == {"medium"}
    assert set(long.loc[long["run_id"] == "run_2", "severity"]) == {"low"}
    assert long.loc[~long["detected"], "delay_s"].isna().all()
    assert long.loc[long["detected"], "delay_s"].notna().all()
    assert (long.groupby(["run_id", "fault_id", "method"]).size() == 1).all()


def test_pooled_comparisons_equal_the_sum_over_runs():
    methods = ["rules", "rules_tuned", "random_forest"]
    frames = [delays_frame(6, methods, seed=s) for s in (1, 2, 3)]
    runs = [fake_run(f"run_{i}", {m: metrics(0.5) for m in methods}, f) for i, f in enumerate(frames, 1)]
    comp = pooled_comparisons(pooled_wide(runs), methods).set_index(["method", "rule"])
    assert list(comp.index) == [("random_forest", "rules"), ("random_forest", "rules_tuned")]
    for rule in ("rules", "rules_tuned"):
        per_run = [compare_with_rules(pd.DataFrame(f), "random_forest", rule=rule) for f in frames]
        for key in ("earlier", "same", "later_or_missed", "both_missed"):
            assert comp.at[("random_forest", rule), key] == sum(c[key] for c in per_run)
        assert comp.at[("random_forest", rule), "faults"] == 18


# --- loading ---------------------------------------------------------------------------
def test_expand_runs_globs_and_natural_order(tmp_path):
    for name in ("run_10", "run_2", "run_1", "other"):
        (tmp_path / name).mkdir()
    (tmp_path / "run_3.txt").write_text("not a folder")
    paths = expand_runs([str(tmp_path / "run_*")])
    assert [p.name for p in paths] == ["run_1", "run_2", "run_10"]
    assert [p.name for p in expand_runs([str(tmp_path / "run_2"), str(tmp_path / "run_2")])] == ["run_2"]


def test_incomplete_or_missing_runs_give_clear_errors(tmp_path):
    (tmp_path / "run_1").mkdir()
    with pytest.raises(ValueError, match="results.json is missing"):
        load_run(tmp_path / "run_1")
    with pytest.raises(ValueError, match="no run folders"):
        load_runs([str(tmp_path / "nothing_*")])


# --- end to end on synthetic runs ------------------------------------------------------
@pytest.fixture(scope="module")
def evaluated_runs(synthetic_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    shutil.copytree(synthetic_run, root / "run_1")
    generate(root / "run_2", duration_min=45, warmup_min=4, rps=4.0, seed=4, min_fault_s=60, max_fault_s=100,
             min_gap_s=120, max_gap_s=200, start=dt.datetime(2026, 3, 2, 12, 0, tzinfo=dt.timezone.utc))
    for run in ("run_1", "run_2"):
        evaluate_paths(root / run / "logs", root / run / "labels.csv", out_dir=root / run / "detector",
                       make_figures=False)
    return root


def test_aggregate_writes_every_output(evaluated_runs, tmp_path):
    out = tmp_path / "summary"
    result = aggregate([str(evaluated_runs / "run_*")], out)
    for name in ("summary_metrics.csv", "summary_metrics.md", "pooled_delays.csv", "summary_q3.md",
                 "f1_by_method.png", "pooled_delays.png", "detection_rate_by_type.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert result["runs"] == ["run_1", "run_2"]
    assert result["methods"] == list(METHOD_ORDER)

    summary = pd.read_csv(out / "summary_metrics.csv")
    assert list(summary["method"]) == list(METHOD_ORDER)
    assert (summary["runs"] == 2).all()
    f1 = [json.loads((evaluated_runs / r / "detector" / "results.json").read_text())["metrics"]["pca"]["f1"]
          for r in ("run_1", "run_2")]
    assert summary.set_index("method").at["pca", "f1_mean"] == pytest.approx(np.mean(f1))

    pooled = pd.read_csv(out / "pooled_delays.csv")
    faults = sum(json.loads((evaluated_runs / r / "detector" / "results.json").read_text())["dataset"]["faults_test"]
                 for r in ("run_1", "run_2"))
    assert len(pooled) == faults * len(METHOD_ORDER)
    assert set(pooled["severity"]) <= {"low", "medium", "high"}
    assert {"run_id", "fault_id", "type", "severity", "service", "method", "detected", "delay_s"} <= set(pooled.columns)

    q3 = (out / "summary_q3.md").read_text()
    for heading in ("ML compared with each rule variant", "## By fault type", "## By severity",
                    "by fault type", "by severity"):
        assert heading in q3

    from PIL import Image
    with Image.open(out / "f1_by_method.png") as img:
        assert round(img.info["dpi"][0]) == 300


def test_cli_aggregate(evaluated_runs, tmp_path, capsys):
    out = tmp_path / "cli_summary"
    code = cli_main(["aggregate", "--runs", str(evaluated_runs / "run_1"), str(evaluated_runs / "run_2"),
                     "--out", str(out), "--no-figures"])
    assert code == 0
    assert "Detection quality over 2 runs" in capsys.readouterr().out
    assert not (out / "f1_by_method.png").exists()
    code = cli_main(["aggregate", "--runs", str(tmp_path / "missing_*"), "--out", str(out)])
    assert code == 2


def test_runs_with_different_evaluation_settings_are_refused():
    same = {"window_s": 30.0, "step_s": 10.0, "train_frac": 0.6, "grace_s": 15.0}
    a = Run("run_1", Path("run_1"), {"metrics": {}, "config": same}, pd.DataFrame())
    b = Run("run_2", Path("run_2"), {"metrics": {}, "config": {**same, "window_s": 60.0}}, pd.DataFrame())
    check_same_settings([a, a])
    with pytest.raises(ValueError, match="window_s: 30.0, 60.0"):
        check_same_settings([a, b])
