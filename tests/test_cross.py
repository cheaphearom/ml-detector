"""Cross-run generalisation: train on one whole run, test on another (``detector.cli cross``)."""
import json
import shutil

import pandas as pd
import pytest

from detector.cli import main as cli_main
from detector.evaluate import cross_paths, evaluate_cross, evaluate_paths, load_run_data
from detector.models import METHOD_ORDER


@pytest.fixture(scope="module")
def runs(synthetic_run, second_synthetic_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("cross")
    shutil.copytree(synthetic_run, root / "run_1")
    shutil.copytree(second_synthetic_run, root / "run_2")
    return root


def test_cross_writes_every_output_and_tests_the_whole_second_run(runs, tmp_path):
    out = tmp_path / "cross"
    res = cross_paths(runs / "run_1", runs / "run_2", out_dir=out)
    for name in ("results.json", "report.md", "detection_delays.csv", "window_predictions.csv", "timeline.png",
                 "metrics.png", "detection_delay.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    saved = json.loads((out / "results.json").read_text())
    assert set(saved["metrics"]) == set(METHOD_ORDER)
    assert saved["config"]["mode"] == "cross"
    assert saved["dataset"]["cross"]["train_run"] == "run_1" and saved["dataset"]["cross"]["test_run"] == "run_2"
    assert saved["dataset"]["windows_dropped_at_split"] == 0

    # every fault of the test run is evaluated, none of the training run
    _, labels2, _ = load_run_data(runs / "run_2" / "logs", runs / "run_2" / "labels.csv")
    delays = pd.read_csv(out / "detection_delays.csv", dtype={"fault_id": str})
    windows = pd.read_csv(out / "window_predictions.csv")
    first = windows["window_start"].min()
    expected = labels2[[t.timestamp() >= first for t in labels2["start"]]]
    assert sorted(delays["fault_id"]) == sorted(expected["fault_id"].astype(str))
    assert len(windows) == saved["dataset"]["windows_test"]
    assert res["comparison_vs_rule_variants"]

    report = (out / "report.md").read_text()
    assert "Cross-run evaluation report" in report and "**whole** run `run_1`" in report
    assert "Cross-run vs within-run" not in report          # run_2 has no own evaluation yet


def test_nothing_of_the_test_run_is_used_for_fitting(runs):
    tr_logs, tr_labels, _ = load_run_data(runs / "run_1" / "logs", runs / "run_1" / "labels.csv")
    te_logs, te_labels, _ = load_run_data(runs / "run_2" / "logs", runs / "run_2" / "labels.csv")
    full = evaluate_cross(tr_logs, tr_labels, te_logs, te_labels, make_figures=False)
    fewer = evaluate_cross(tr_logs, tr_labels, te_logs, te_labels.iloc[: len(te_labels) // 2], make_figures=False)
    for key in ("isolation_forest", "pca"):
        assert full["config"][key]["threshold"] == fewer["config"][key]["threshold"]
    assert full["rule_tuning"]["selected"] == fewer["rule_tuning"]["selected"]
    assert full["dataset"]["known_templates"] == fewer["dataset"]["known_templates"]
    assert full["dataset"]["windows_train"] == fewer["dataset"]["windows_train"]
    assert fewer["dataset"]["faults_test"] < full["dataset"]["faults_test"]


def test_within_run_reference_is_reported_when_available(runs, tmp_path):
    run2 = tmp_path / "run_2"
    shutil.copytree(runs / "run_2", run2)
    own = evaluate_paths(run2 / "logs", run2 / "labels.csv", out_dir=run2 / "detector", make_figures=False)
    res = cross_paths(runs / "run_1", run2, out_dir=tmp_path / "out", make_figures=False)
    ref = res["within_run_reference"]
    assert ref["methods"]["random_forest"]["f1"] == own["metrics"]["random_forest"]["f1"]
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Cross-run vs within-run" in report and "Δ F1" in report
    # an evaluation with another window length is not comparable and is ignored
    other = cross_paths(runs / "run_1", run2, window=60.0, make_figures=False)
    assert other["within_run_reference"] is None


def test_cross_refuses_the_same_run_and_reports_cli_errors(runs, tmp_path, capsys):
    with pytest.raises(ValueError, match="must be different"):
        cross_paths(runs / "run_1", runs / "run_1")
    code = cli_main(["cross", "--train-run", str(runs / "run_1"), "--test-run", str(runs / "run_2"),
                     "--out", str(tmp_path / "cli"), "--no-figures", "--methods", "pca,random_forest"])
    assert code == 0
    saved = json.loads((tmp_path / "cli" / "results.json").read_text())
    assert set(saved["metrics"]) == {"rules", "pca", "random_forest"}   # a subset + the default rules
    assert "Random Forest vs rules" in capsys.readouterr().out
    code = cli_main(["cross", "--train-run", str(runs / "run_1"), "--test-run", str(tmp_path / "missing"),
                     "--out", str(tmp_path / "x")])
    assert code == 2
