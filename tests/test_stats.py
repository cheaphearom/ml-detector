"""Bootstrap confidence intervals, McNemar, Wilcoxon, Holm and the stats report."""
import datetime as dt
import math
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from sklearn.metrics import f1_score

from detector import stats as st
from detector.aggregate import aggregate
from detector.evaluate import evaluate_paths, window_clusters
from detector.parsing import load_labels
from detector.synthetic import generate


# --- Holm ------------------------------------------------------------------------------
def test_holm_matches_the_textbook_procedure():
    adj = st.holm([0.01, 0.04, 0.03, None])
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone: 0.03, 0.06, 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert adj[3] is None
    assert st.holm([0.5, 0.9]) == [1.0, 1.0]
    assert st.holm([float("nan"), 0.02]) == [None, pytest.approx(0.02)]


# --- McNemar ---------------------------------------------------------------------------
def test_mcnemar_exact_uses_only_discordant_faults():
    ml = [True] * 8 + [True] * 10 + [False] * 4
    rule = [False] * 8 + [True] * 10 + [False] * 4
    res = st.mcnemar_exact(ml, rule)
    assert (res["ml_only"], res["rule_only"], res["n"], res["n_faults"]) == (8, 0, 8, 22)
    assert res["p"] == pytest.approx(2 * 0.5 ** 8)
    assert res["p"] == pytest.approx(sps.binomtest(8, 8, 0.5).pvalue)


def test_mcnemar_refuses_too_few_discordant_faults():
    res = st.mcnemar_exact([True] * 5 + [True] * 20, [False] * 5 + [True] * 20)
    assert res["p"] is None and "only 5 faults" in res["note"]


# --- Wilcoxon --------------------------------------------------------------------------
def test_wilcoxon_drops_ties_and_reports_effect_size():
    rule = np.array([40, 50, 30, 60, 45, 50, 20, 30, 30, 10], dtype=float)
    ml = np.array([10, 20, 20, 30, 15, 20, 20, 30, 10, 20], dtype=float)
    res = st.wilcoxon_delays(ml, rule)
    diff = rule - ml
    assert res["n_pairs"] == 10 and res["n"] == int((diff != 0).sum()) == 8
    assert res["median_diff_s"] == pytest.approx(np.median(diff))       # positive = ML earlier
    with pytest.warns(UserWarning):          # scipy warns that n = 8 is small for the approximation
        ref = sps.wilcoxon(diff[diff != 0], correction=False, method="approx")
    assert res["p"] == pytest.approx(ref.pvalue)
    assert res["r"] == pytest.approx(abs(ref.zstatistic) / math.sqrt(8))


def test_wilcoxon_refuses_fewer_than_six_nonzero_pairs():
    res = st.wilcoxon_delays([10, 20, 30, 40, 50, 60, 70], [20, 20, 40, 50, 60, 70, 70])
    assert res["n"] == 5 and res["p"] is None and "need ≥ 6" in res["note"]


# --- bootstrap -------------------------------------------------------------------------
def test_stratified_indices_stay_inside_their_stratum_and_are_reproducible():
    strata = np.array(["a"] * 3 + ["b"] * 5)
    idx = st.stratified_indices(strata, n_boot=50, seed=1)
    assert idx.shape == (50, 8)
    assert (strata[idx[:, :3]] == "a").all() and (strata[idx[:, 3:]] == "b").all()
    assert (idx == st.stratified_indices(strata, n_boot=50, seed=1)).all()


def cluster_windows(seed=0, n_clusters=12, size=10):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        run = f"run_{c % 3}"
        y = np.r_[np.ones(3, int), np.zeros(size - 3, int)]
        good = rng.random() < 0.7                               # cluster-level quality: windows move together
        pred = y.copy() if good else np.zeros(size, int)
        rows.append(pd.DataFrame({"run_id": run, "cluster": f"F{c}", "label": y, "m_pred": pred,
                                  "perfect_pred": y}))
    return pd.concat(rows, ignore_index=True)


def test_f1_cluster_ci_point_estimate_and_interval():
    w = cluster_windows()
    ci = st.f1_cluster_ci(w, ["m", "perfect", "absent"]).set_index("method")
    assert "absent" not in ci.index
    assert ci.at["m", "f1_pooled"] == pytest.approx(f1_score(w["label"], w["m_pred"]))
    assert ci.at["m", "f1_ci_low"] <= ci.at["m", "f1_pooled"] <= ci.at["m", "f1_ci_high"]
    assert ci.at["m", "f1_ci_low"] < ci.at["m", "f1_ci_high"]
    assert ci.at["m", "clusters"] == 12 and ci.at["m", "windows"] == 120
    assert (ci.at["perfect", "f1_ci_low"], ci.at["perfect", "f1_ci_high"]) == (1.0, 1.0)
    again = st.f1_cluster_ci(w, ["m"]).set_index("method")
    assert again.at["m", "f1_ci_low"] == ci.at["m", "f1_ci_low"]          # fixed seed


def test_f1_run_ci_fallback():
    per_run = pd.DataFrame({"method": ["m"] * 3 + ["one"], "f1": [0.6, 0.7, 0.8, 0.5]})
    ci = st.f1_run_ci(per_run, ["m", "one"]).set_index("method")
    assert ci.at["m", "f1_pooled"] == pytest.approx(0.7)
    assert 0.6 - 1e-9 <= ci.at["m", "f1_ci_low"] <= 0.7 <= ci.at["m", "f1_ci_high"] <= 0.8 + 1e-9
    assert math.isnan(ci.at["one", "f1_ci_low"])                         # one run: no interval


def wide_frame():
    return pd.DataFrame({
        "run_id": ["run_1"] * 6 + ["run_2"] * 6,
        "rules_tuned_detected": [True] * 10 + [False] * 2,
        "rules_tuned_delay_s": [30.0, 40, 30, 50, 30, 20, 40, 30, 50, 30, np.nan, np.nan],
        "rules_tuned_k1_detected": [True] * 10 + [False] * 2,
        "rules_tuned_k1_delay_s": [30.0, 40, 30, 50, 30, 20, 40, 30, 50, 30, np.nan, np.nan],
        "pca_detected": [True] * 12,
        "pca_delay_s": [10.0, 10, 20, 20, 10, 20, 10, 10, 20, 20, 30, 30],
    })


def test_delay_ci():
    ci = st.delay_ci(wide_frame(), ["rules_tuned", "pca", "absent"]).set_index("method")
    assert ci.at["pca", "mean_delay_s"] == pytest.approx(np.mean(wide_frame()["pca_delay_s"]))
    assert ci.at["rules_tuned", "detected"] == 10 and ci.at["rules_tuned", "faults"] == 12
    for m in ("pca", "rules_tuned"):
        assert ci.at[m, "delay_ci_low"] <= ci.at[m, "mean_delay_s"] <= ci.at[m, "delay_ci_high"]


def test_paired_tests_skip_identical_variants_and_apply_holm():
    tests = st.paired_tests(wide_frame(), ["rules_tuned", "rules_tuned_k1", "pca"])
    assert tests.attrs["identical"] == {"rules_tuned_k1": "rules_tuned"}
    assert set(tests["rule"]) == {"rules_tuned"}
    delay = tests[tests["hypothesis"] == "delay"].iloc[0]
    assert delay["n_pairs"] == 10 and delay["median_diff_s"] > 0
    assert delay["p_holm"] == pytest.approx(min(1.0, delay["p"] * tests["p"].notna().sum()))
    detection = tests[tests["hypothesis"] == "detection"].iloc[0]
    assert pd.isna(detection["p"])                                       # only 2 discordant faults
    line = st.hypothesis_line(delay)
    assert line.startswith("H3₀ (no difference in detection delay) rejected for PCA (reconstruction) vs rules_tuned")
    assert "earlier" in line and "n = 9 faults with different delays of 10 detected by both" in line
    assert "not tested" in st.hypothesis_line(detection)
    report = st.render_stats(pd.DataFrame(columns=["method"]), pd.DataFrame(columns=["method"]), tests, 2)
    assert "are not repeated" in report and "H3₀" in report


# --- windows and end to end ------------------------------------------------------------
def test_window_clusters_follow_the_latest_fault_start(synthetic_run):
    labels = load_labels(synthetic_run / "labels.csv").sort_values("start")
    first = labels["start"].iloc[0].timestamp()
    test = pd.DataFrame({"window_start": [first - 100, first - 5, first + 50],
                         "window_end": [first - 70, first + 25, first + 80]})
    clusters = window_clusters(test, labels)
    assert clusters[0] == "pre"
    assert clusters[1] == clusters[2] == str(labels["fault_id"].iloc[0])


@pytest.fixture(scope="module")
def two_runs(synthetic_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("stats_runs")
    shutil.copytree(synthetic_run, root / "run_1")
    generate(root / "run_2", duration_min=45, warmup_min=4, rps=4.0, seed=5, min_fault_s=60, max_fault_s=100,
             min_gap_s=120, max_gap_s=200, start=dt.datetime(2026, 3, 3, 12, 0, tzinfo=dt.timezone.utc))
    for run in ("run_1", "run_2"):
        evaluate_paths(root / run / "logs", root / run / "labels.csv", out_dir=root / run / "detector",
                       make_figures=False)
    return root


def test_evaluate_writes_window_predictions(two_runs):
    import json
    w = pd.read_csv(two_runs / "run_1" / "detector" / "window_predictions.csv")
    res = json.loads((two_runs / "run_1" / "detector" / "results.json").read_text())
    assert len(w) == res["dataset"]["windows_test"]
    assert {"label", "cluster", "pca_pred", "rules_tuned_score"} <= set(w.columns)
    m = res["metrics"]["random_forest"]
    assert int(((w["random_forest_pred"] == 1) & (w["label"] == 1)).sum()) == m["tp"]


def test_aggregate_writes_stats_and_falls_back_without_window_files(two_runs, tmp_path):
    out = tmp_path / "s"
    result = aggregate([str(two_runs / "run_*")], out, make_figures=False, n_boot=200)
    for name in ("stats.md", "stats_tests.csv", "ci_f1.csv", "ci_delay.csv"):
        assert (out / name).exists(), name
    assert "cluster bootstrap by fault" in result["f1_ci"]["ci_method"].iloc[0]
    text = (out / "stats.md").read_text()
    assert "## Hypotheses" in text and "Holm" in text

    shutil.copytree(two_runs, tmp_path / "old")
    (tmp_path / "old" / "run_2" / "detector" / "window_predictions.csv").unlink()
    old = aggregate([str(tmp_path / "old" / "run_*")], tmp_path / "s2", make_figures=False, n_boot=200)
    assert "bootstrap over runs" in old["f1_ci"]["ci_method"].iloc[0]
