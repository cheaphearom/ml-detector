"""Offline evaluation: rule variants vs Isolation Forest, Random Forest, PCA and Logistic Regression.

Pipeline
  1. parse logs, build 30 s / 10 s sliding windows, label windows that overlap
     [fault start, fault end + grace]
  2. chronological split: windows ending before ``t_split`` train, windows starting after it
     test; windows that cross the split are dropped (no leakage through overlapping windows)
  3. "unseen templates" are counted against the templates of the normal training logs
  4. fit the detectors on the training windows (the tuned rule variants choose their thresholds
     on the training windows only), score the test windows
  5. metrics (Q1/Q2), per-fault detection delay and comparison with the rules (Q3)

Outputs in ``out_dir``: results.json, report.md, detection_delays.csv, window_predictions.csv,
timeline.png, metrics.png, detection_delay.png.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .profiles import get_profile
from .features import (
    add_unseen_templates, build_windows, feature_name, in_fault_mask, known_templates,
    label_windows,
)
from .models import (
    METHOD_LABELS, METHOD_ORDER, ML_METHODS, RULE_METHODS, IsolationForestDetector, LogisticRegressionDetector,
    PCADetector, RandomForestDetector, RuleBasedDetector, TunedRuleBasedDetector, TunedRuleK1Detector,
    alert_episodes,
)
from .parsing import load_experiment_meta, load_labels, load_logs

# Reference palette, categorical slots in fixed order per method (colour follows the method).
COLORS = {
    RuleBasedDetector.name: "#2a78d6",
    IsolationForestDetector.name: "#eb6834",
    RandomForestDetector.name: "#1baf7a",
    TunedRuleBasedDetector.name: "#eda100",
    TunedRuleK1Detector.name: "#6250d6",
    PCADetector.name: "#008300",
    LogisticRegressionDetector.name: "#e87ba4",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FAULT_SHADE = "#d6d5d0"


@dataclass
class Dataset:
    windows: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    labels: pd.DataFrame
    t_split: float
    known_templates: set[str]
    window: float
    step: float
    grace: float
    meta: dict = field(default_factory=dict)


def prepare_dataset(logs: pd.DataFrame, labels: pd.DataFrame, window: float = 30.0, step: float = 10.0,
                    train_frac: float = 0.6, grace: float = 15.0,
                    start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> Dataset:
    """Builds labelled windows and the chronological train/test split."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if start is not None:
        logs = logs[logs["ts"] >= start]
    if end is not None:
        logs = logs[logs["ts"] <= end]
    logs = logs.reset_index(drop=True)
    if logs.empty:
        raise ValueError("no log records in the selected period")

    windows = build_windows(logs, window=window, step=step)
    if len(windows) < 20:
        raise ValueError(f"only {len(windows)} windows - the log period is too short to evaluate")
    windows = label_windows(windows, labels, grace=grace)

    t0 = float(windows["window_start"].min())
    t1 = float(windows["window_end"].max())
    t_split = t0 + train_frac * (t1 - t0)

    normal_train_logs = (logs["t"].to_numpy() < t_split) & ~in_fault_mask(logs["t"].to_numpy(), labels, grace)
    known = known_templates(logs, normal_train_logs)
    windows = add_unseen_templates(windows, logs, known)

    train = windows[windows["window_end"] <= t_split].reset_index(drop=True)
    test = windows[windows["window_start"] >= t_split].reset_index(drop=True)
    return Dataset(windows=windows, train=train, test=test, labels=labels, t_split=t_split,
                   known_templates=known, window=window, step=step, grace=grace)


def select_methods(methods=None, tune_rules: bool = True) -> list[str]:
    """Methods to evaluate, in :data:`METHOD_ORDER`.

    ``methods=None`` means every method. The default rules are always included (every
    comparison is made against them); ``tune_rules=False`` removes the tuned rule variants.
    """
    wanted = set(METHOD_ORDER if methods is None else methods)
    unknown = wanted - set(METHOD_ORDER)
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(sorted(unknown))}; choose from {', '.join(METHOD_ORDER)}")
    wanted.add(RuleBasedDetector.name)
    if not tune_rules:
        wanted -= {TunedRuleBasedDetector.name, TunedRuleK1Detector.name}
    return [m for m in METHOD_ORDER if m in wanted]


def make_detectors(step: float, seed: int, tune_rules: bool = True, rule_grid: dict | None = None,
                   methods=None) -> dict:
    """The compared detectors, in :data:`METHOD_ORDER` (see :func:`select_methods`)."""
    factories = {
        RuleBasedDetector.name: lambda: RuleBasedDetector(step=step),
        TunedRuleBasedDetector.name: lambda: TunedRuleBasedDetector(step=step, grid=rule_grid),
        TunedRuleK1Detector.name: lambda: TunedRuleK1Detector(step=step, grid=rule_grid),
        IsolationForestDetector.name: lambda: IsolationForestDetector(random_state=seed),
        RandomForestDetector.name: lambda: RandomForestDetector(random_state=seed),
        PCADetector.name: lambda: PCADetector(percentile=IsolationForestDetector().percentile),
        LogisticRegressionDetector.name: lambda: LogisticRegressionDetector(random_state=seed),
    }
    return {m: factories[m]() for m in select_methods(methods, tune_rules)}


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray,
                           starts: np.ndarray, step: float, hours: float) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = None
    if np.unique(y).size == 2 and np.all(np.isfinite(score)):
        auc = float(roc_auc_score(y, score))
    episodes = alert_episodes(pred, starts, step)
    false_episodes = sum(1 for a, b in episodes if y[a:b + 1].sum() == 0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": auc,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "alert_episodes": len(episodes),
        "false_alarm_episodes": int(false_episodes),
        "false_alarms_per_hour": float(false_episodes / hours) if hours > 0 else 0.0,
    }


def detection_delays(test: pd.DataFrame, labels: pd.DataFrame, preds: dict[str, np.ndarray],
                     t_split: float, grace: float) -> pd.DataFrame:
    """Per fault in the test period: delay = end of the first alerting window - fault start."""
    ws = test["window_start"].to_numpy()
    we = test["window_end"].to_numpy()
    rows = []
    for f in labels.itertuples(index=False):
        s = f.start.timestamp()
        e = f.end.timestamp()
        if s < t_split or s > we.max(initial=-math.inf):
            continue
        in_scope = (we > s) & (ws <= e + grace)
        row = {
            "fault_id": f.fault_id, "service": f.service, "type": f.type,
            "start": f.start.isoformat(), "end": f.end.isoformat(),
            "duration_s": round(e - s, 3), "params": f.params,
        }
        for name, pred in preds.items():
            hits = np.flatnonzero(in_scope & (pred == 1))
            if hits.size:
                first = hits[np.argmin(we[hits])]
                row[f"{name}_detected"] = True
                row[f"{name}_delay_s"] = round(float(we[first] - s), 3)
            else:
                row[f"{name}_detected"] = False
                row[f"{name}_delay_s"] = np.nan
        rows.append(row)
    columns = ["fault_id", "service", "type", "start", "end", "duration_s", "params"]
    for name in preds:
        columns += [f"{name}_detected", f"{name}_delay_s"]
    return pd.DataFrame(rows, columns=columns)


def window_clusters(test: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    """Cluster of each test window for the cluster bootstrap: the most recent fault that started
    before the window ended (the fault plus the normal gap after it). Windows before the first
    fault form the cluster ``"pre"``."""
    we = test["window_end"].to_numpy(dtype=float)
    if labels.empty:
        return np.full(len(test), "pre", dtype=object)
    lab = labels.sort_values("start")
    starts = np.array([t.timestamp() for t in lab["start"]])
    ids = lab["fault_id"].astype(str).to_numpy()
    pos = np.searchsorted(starts, we, side="left") - 1
    return np.where(pos >= 0, ids[np.clip(pos, 0, None)], "pre").astype(object)


def window_predictions(test: pd.DataFrame, labels: pd.DataFrame, scores: dict, preds: dict) -> pd.DataFrame:
    """Per test window: label, fault, bootstrap cluster and every method's score and alert."""
    out = test[["window_start", "window_end", "label", "fault_id"]].copy()
    out["cluster"] = window_clusters(test, labels)
    for m in scores:
        out[f"{m}_score"] = scores[m]
        out[f"{m}_pred"] = preds[m]
    return out


def delay_summary(delays: pd.DataFrame, method: str) -> dict:
    n = len(delays)
    if n == 0:
        return {"faults": 0, "detected": 0, "detection_rate": None,
                "mean_delay_s": None, "median_delay_s": None, "max_delay_s": None}
    d = delays.loc[delays[f"{method}_detected"], f"{method}_delay_s"]
    return {
        "faults": int(n),
        "detected": int(d.size),
        "detection_rate": float(d.size / n),
        "mean_delay_s": float(d.mean()) if d.size else None,
        "median_delay_s": float(d.median()) if d.size else None,
        "max_delay_s": float(d.max()) if d.size else None,
    }


def compare_with_rules(delays: pd.DataFrame, method: str, eps: float = 1e-6, rule: str = "rules") -> dict:
    """Per fault: did ``method`` alert earlier than, at the same time as, or later than ``rule``?"""
    counts = {"earlier": 0, "same": 0, "later_or_missed": 0, "both_missed": 0}
    saved = []
    for _, row in delays.iterrows():
        ml_hit, rule_hit = bool(row[f"{method}_detected"]), bool(row[f"{rule}_detected"])
        ml_d, rule_d = row[f"{method}_delay_s"], row[f"{rule}_delay_s"]
        if not ml_hit and not rule_hit:
            counts["both_missed"] += 1
        elif ml_hit and not rule_hit:
            counts["earlier"] += 1
        elif rule_hit and not ml_hit:
            counts["later_or_missed"] += 1
        else:
            saved.append(rule_d - ml_d)
            if ml_d < rule_d - eps:
                counts["earlier"] += 1
            elif abs(ml_d - rule_d) <= eps:
                counts["same"] += 1
            else:
                counts["later_or_missed"] += 1
    return {
        **counts,
        "faults_detected_by_both": len(saved),
        "mean_time_saved_s": float(np.mean(saved)) if saved else None,
        "median_time_saved_s": float(np.median(saved)) if saved else None,
        "detected_only_by_ml": int(sum(1 for _, r in delays.iterrows()
                                       if r[f"{method}_detected"] and not r[f"{rule}_detected"])),
    }


def breakdown(delays: pd.DataFrame, by: str, methods: list[str]) -> dict:
    result = {}
    for key, group in delays.groupby(by, sort=True):
        result[str(key)] = {"faults": int(len(group))}
        for m in methods:
            result[str(key)][m] = delay_summary(group, m)
    return result


# ---------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------
def evaluate(logs: pd.DataFrame, labels: pd.DataFrame, window: float = 30.0, step: float = 10.0,
             train_frac: float = 0.6, grace: float = 15.0, seed: int = 42,
             start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
             out_dir: str | os.PathLike | None = None, meta: dict | None = None,
             make_figures: bool = True, tune_rules: bool = True, rule_grid: dict | None = None,
             methods=None) -> dict:
    """Runs the full comparison and (optionally) writes every output file to ``out_dir``."""
    if labels.empty:
        raise ValueError("labels are empty - the evaluation needs at least one injected fault")
    ds = prepare_dataset(logs, labels, window, step, train_frac, grace, start, end)
    dataset = {
        "log_records": int(len(logs)),
        "windows_total": int(len(ds.windows)),
        "windows_dropped_at_split": int(len(ds.windows) - len(ds.train) - len(ds.test)),
        "faults_total": int(len(labels)),
        "known_templates": len(ds.known_templates),
        "period_start": pd.Timestamp(ds.windows["window_start"].min(), unit="s", tz="UTC").isoformat(),
        "split_at": pd.Timestamp(ds.t_split, unit="s", tz="UTC").isoformat(),
        "period_end": pd.Timestamp(ds.windows["window_end"].max(), unit="s", tz="UTC").isoformat(),
        "experiment": meta or {},
    }
    config = {"window_s": window, "step_s": step, "train_frac": train_frac, "grace_s": grace, "seed": seed}
    return run_comparison(ds.train, ds.test, labels, ds.t_split, step, grace, seed, config, dataset,
                          out_dir=out_dir, make_figures=make_figures, tune_rules=tune_rules,
                          rule_grid=rule_grid, methods=methods)


def run_comparison(train: pd.DataFrame, test: pd.DataFrame, test_labels: pd.DataFrame, t_split: float,
                   step: float, grace: float, seed: int, config: dict, dataset: dict,
                   out_dir: str | os.PathLike | None = None, make_figures: bool = True,
                   tune_rules: bool = True, rule_grid: dict | None = None, methods=None,
                   extra_results: dict | None = None) -> dict:
    """Fits every detector on ``train``, scores ``test`` and builds metrics, delays and outputs.

    Shared by :func:`evaluate` (chronological split of one run) and :func:`evaluate_cross`
    (one run trains, another run tests). Delays are measured for faults of ``test_labels`` that
    start at or after ``t_split``. ``extra_results`` are added to ``results.json``.
    """
    if train["label"].sum() == 0:
        raise ValueError("no anomalous windows in the training period - run a longer experiment "
                         "or increase --train-frac")
    if len(test) == 0:
        raise ValueError("no test windows - decrease --train-frac")

    detectors = make_detectors(step, seed, tune_rules=tune_rules, rule_grid=rule_grid, methods=methods)
    y_train = train["label"].to_numpy()
    for det in detectors.values():
        det.fit(train, y_train)

    y_test = test["label"].to_numpy()
    starts = test["window_start"].to_numpy()
    test_hours = (test["window_end"].max() - test["window_start"].min()) / 3600.0
    scores, preds, metrics = {}, {}, {}
    for name, det in detectors.items():
        scores[name] = det.score(test)
        preds[name] = det.predict(test)
        metrics[name] = classification_metrics(y_test, preds[name], scores[name], starts, step, test_hours)

    delays = detection_delays(test, test_labels, preds, t_split, grace)
    methods = list(detectors)
    delay_stats = {m: delay_summary(delays, m) for m in methods}
    ml_methods = [m for m in methods if m in ML_METHODS]
    rule_methods = [m for m in methods if m in RULE_METHODS]
    comparison = {m: compare_with_rules(delays, m) for m in ml_methods}
    comparison_variants = {r: {m: compare_with_rules(delays, m, rule=r) for m in ml_methods}
                           for r in rule_methods}

    results = {
        "config": {
            **config,
            "methods": list(detectors),
            **{r: vars(detectors[r].config) | {} for r in rule_methods},
            **{m: model_config(detectors[m]) for m in ml_methods},
        },
        "dataset": {
            **dataset,
            "windows_train": int(len(train)),
            "windows_test": int(len(test)),
            "anomalous_windows_train": int(train["label"].sum()),
            "anomalous_windows_test": int(test["label"].sum()),
            "faults_test": int(len(delays)),
            "test_hours": float(test_hours),
        },
        "metrics": metrics,
        "detection_delay": delay_stats,
        "comparison_vs_rules": comparison,
        "comparison_vs_rule_variants": comparison_variants,
        "rule_tuning": (detectors[TunedRuleBasedDetector.name].tuning_
                        if TunedRuleBasedDetector.name in detectors else None),
        "by_fault_type": breakdown(delays, "type", methods) if len(delays) else {},
        "by_service": breakdown(delays, "service", methods) if len(delays) else {},
        "feature_importance": ([{"feature": f, "importance": v}
                                for f, v in detectors[RandomForestDetector.name].feature_importance(top=20)]
                               if RandomForestDetector.name in detectors else []),
        "logistic_regression_coefficients": (
            [{"feature": f, "coefficient": v}
             for f, v in detectors[LogisticRegressionDetector.name].coefficients(top=20)]
            if LogisticRegressionDetector.name in detectors else []),
    }
    results.update(extra_results or {})

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
        delays.to_csv(out / "detection_delays.csv", index=False)
        window_predictions(test, test_labels, scores, preds).to_csv(out / "window_predictions.csv", index=False)
        (out / "report.md").write_text(render_report(results, delays), encoding="utf-8")
        if make_figures:
            plot_timeline(test, test_labels, preds, t_split, grace, out / "timeline.png")
            plot_metrics(metrics, out / "metrics.png")
            plot_delays(delays, methods, out / "detection_delay.png")
        results["outputs"] = sorted(p.name for p in out.iterdir() if p.is_file())
    return results


def model_config(det) -> dict:
    """Settings and learned thresholds of an ML detector, for ``results.json``."""
    if isinstance(det, IsolationForestDetector):
        return {"n_estimators": det.n_estimators, "percentile": det.percentile, "threshold": det.threshold_,
                "features": det.columns_}
    if isinstance(det, RandomForestDetector):
        return {"n_estimators": det.n_estimators, "class_weight": "balanced", "threshold": det.threshold}
    if isinstance(det, PCADetector):
        return {"variance": det.variance, "n_components": det.n_components_,
                "explained_variance": det.explained_variance_, "percentile": det.percentile,
                "threshold": det.threshold_, "features": det.columns_}
    if isinstance(det, LogisticRegressionDetector):
        return {"C": det.C, "class_weight": "balanced", "threshold": det.threshold,
                "preprocessing": "log1p + StandardScaler", "features": len(det.columns_)}
    return {}


def load_run_data(logs_path: str | os.PathLike, labels_path: str | os.PathLike, start: str | None = None,
                  end: str | None = None, allow_incomplete: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Loads logs, labels and ``experiment.json`` of one run (see :func:`evaluate_paths`)."""
    if not Path(labels_path).exists():
        raise ValueError(f"labels file not found: {labels_path} - check the path; if the experiment is "
                         f"still running, it has not finished its first fault yet")
    labels = load_labels(labels_path)
    meta = load_experiment_meta(labels_path)
    if not allow_incomplete:
        if not meta:
            raise ValueError(f"experiment.json is missing next to {labels_path}: the experiment is probably "
                             f"still running. Wait until it logs 'experiment finished' (docker ps / "
                             f"docker logs), or pass --allow-incomplete to evaluate the faults so far")
        if meta.get("completed") is False:
            raise ValueError("this experiment was stopped before it finished, so its results are incomplete. "
                             "Run it again, or pass --allow-incomplete to evaluate it anyway")
    start_ts = pd.Timestamp(start) if start else (pd.Timestamp(meta["load_start"]) if meta.get("load_start") else None)
    end_ts = pd.Timestamp(end) if end else (pd.Timestamp(meta["load_end"]) if meta.get("load_end") else None)
    logs = load_logs(logs_path, start=_utc(start_ts), end=_utc(end_ts))
    if logs.empty:
        raise ValueError(f"no log records in {logs_path} for the selected period")
    return logs, labels, meta


def evaluate_paths(logs_path: str | os.PathLike, labels_path: str | os.PathLike, out_dir=None,
                   window: float = 30.0, step: float = 10.0, train_frac: float = 0.6, grace: float = 15.0,
                   seed: int = 42, start: str | None = None, end: str | None = None,
                   make_figures: bool = True, allow_incomplete: bool = False, tune_rules: bool = True,
                   rule_grid: dict | None = None, methods=None) -> dict:
    """Loads logs and labels from disk and runs :func:`evaluate`.

    If ``experiment.json`` lies next to ``labels.csv``, the evaluation period defaults to the
    time the load generator ran, which keeps service start-up logs out of the data.

    Unless ``allow_incomplete`` is set, an experiment that is still running (no
    ``experiment.json`` yet) or was interrupted (``"completed": false``) is refused, because
    its labels do not cover the whole period.
    """
    logs, labels, meta = load_run_data(logs_path, labels_path, start, end, allow_incomplete)
    return evaluate(logs, labels, window=window, step=step, train_frac=train_frac, grace=grace, seed=seed,
                    out_dir=out_dir, meta=meta, make_figures=make_figures, tune_rules=tune_rules,
                    rule_grid=rule_grid, methods=methods)


# ---------------------------------------------------------------------------------------
# Cross-run generalisation: train on one whole run, test on another whole run
# ---------------------------------------------------------------------------------------
def run_windows(logs: pd.DataFrame, labels: pd.DataFrame, window: float, step: float, grace: float) -> pd.DataFrame:
    windows = build_windows(logs, window=window, step=step)
    if len(windows) < 20:
        raise ValueError(f"only {len(windows)} windows - the log period is too short to evaluate")
    return label_windows(windows, labels, grace=grace)


def evaluate_cross(train_logs: pd.DataFrame, train_labels: pd.DataFrame, test_logs: pd.DataFrame,
                   test_labels: pd.DataFrame, window: float = 30.0, step: float = 10.0, grace: float = 15.0,
                   seed: int = 42, out_dir: str | os.PathLike | None = None, make_figures: bool = True,
                   tune_rules: bool = True, rule_grid: dict | None = None, methods=None,
                   train_name: str = "train run", test_name: str = "test run",
                   train_meta: dict | None = None, test_meta: dict | None = None,
                   within_reference: dict | None = None) -> dict:
    """Trains every method on the whole training run and tests it on the whole test run.

    Nothing of the test run is used for fitting: the known-template vocabulary comes from the
    normal logs of the training run, the rules are tuned on its windows, and all models are fitted
    on its windows. Every fault of the test run is evaluated.
    ``within_reference`` is the test run's own ``results.json`` (chronological split), if available.
    """
    if train_labels.empty or test_labels.empty:
        raise ValueError("both runs need injected faults (labels.csv is empty)")
    train = run_windows(train_logs, train_labels, window, step, grace)
    test = run_windows(test_logs, test_labels, window, step, grace)
    normal_train_logs = ~in_fault_mask(train_logs["t"].to_numpy(), train_labels, grace)
    known = known_templates(train_logs, normal_train_logs)
    train = add_unseen_templates(train, train_logs, known).reset_index(drop=True)
    test = add_unseen_templates(test, test_logs, known).reset_index(drop=True)
    t_test = float(test["window_start"].min())

    def iso(t):
        return pd.Timestamp(t, unit="s", tz="UTC").isoformat()

    dataset = {
        "log_records": int(len(train_logs) + len(test_logs)),
        "windows_total": int(len(train) + len(test)),
        "windows_dropped_at_split": 0,
        "faults_total": int(len(test_labels)),
        "faults_train": int(len(train_labels)),
        "known_templates": len(known),
        "period_start": iso(test["window_start"].min()),
        "split_at": iso(t_test),
        "period_end": iso(test["window_end"].max()),
        "cross": {"train_run": train_name, "test_run": test_name,
                  "train_period": [iso(train["window_start"].min()), iso(train["window_end"].max())],
                  "test_period": [iso(t_test), iso(test["window_end"].max())],
                  "train_experiment": train_meta or {}, "test_experiment": test_meta or {}},
        "experiment": test_meta or {},
    }
    config = {"mode": "cross", "window_s": window, "step_s": step, "train_frac": None, "grace_s": grace,
              "seed": seed, "train_run": train_name, "test_run": test_name}
    extra = {"within_run_reference": _within_reference(within_reference)}
    return run_comparison(train, test, test_labels, t_test, step, grace, seed, config, dataset,
                          out_dir=out_dir, make_figures=make_figures, tune_rules=tune_rules,
                          rule_grid=rule_grid, methods=methods, extra_results=extra)


def _within_reference(ref: dict | None) -> dict | None:
    """F1 and detection rate per method from the test run's own (chronological split) evaluation."""
    if not ref:
        return None
    return {"split_at": ref.get("dataset", {}).get("split_at"),
            "methods": {m: {"f1": r.get("f1"), "roc_auc": r.get("roc_auc"),
                            "false_alarms_per_hour": r.get("false_alarms_per_hour"),
                            "detection_rate": ref.get("detection_delay", {}).get(m, {}).get("detection_rate")}
                        for m, r in ref.get("metrics", {}).items()}}


def cross_paths(train_run: str | os.PathLike, test_run: str | os.PathLike, out_dir=None, window: float = 30.0,
                step: float = 10.0, grace: float = 15.0, seed: int = 42, make_figures: bool = True,
                allow_incomplete: bool = False, tune_rules: bool = True, methods=None,
                logs_subdir: str = "logs", detector_dir: str = "detector") -> dict:
    """Loads two run folders (``logs/``, ``labels.csv``, ``experiment.json``) and runs :func:`evaluate_cross`."""
    train_run, test_run = Path(train_run), Path(test_run)
    if train_run.resolve() == test_run.resolve():
        raise ValueError("--train-run and --test-run must be different runs")
    tr_logs, tr_labels, tr_meta = load_run_data(train_run / logs_subdir, train_run / "labels.csv",
                                                allow_incomplete=allow_incomplete)
    te_logs, te_labels, te_meta = load_run_data(test_run / logs_subdir, test_run / "labels.csv",
                                                allow_incomplete=allow_incomplete)
    ref_path = test_run / detector_dir / "results.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    if ref and ref.get("config", {}).get("window_s") not in (None, window):
        ref = None          # evaluated with another window: not comparable
    return evaluate_cross(tr_logs, tr_labels, te_logs, te_labels, window=window, step=step, grace=grace, seed=seed,
                          out_dir=out_dir, make_figures=make_figures, tune_rules=tune_rules, methods=methods,
                          train_name=train_run.name, test_name=test_run.name, train_meta=tr_meta,
                          test_meta=te_meta, within_reference=ref)


def _utc(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------
def _fmt(v, digits: int = 3, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}{suffix}"
    return f"{v}{suffix}"


def _methods(results: dict) -> list[str]:
    return [m for m in METHOD_ORDER if m in results["metrics"]]


RULE_THRESHOLD_FIELDS = (("gateway_5xx_ratio", "Gateway 5xx ratio >"), ("gateway_p95_ms", "Gateway p95 (ms) >"),
                         ("errors_per_service", "ERROR lines / service >"), ("min_requests", "Min. requests"),
                         ("consecutive", "Consecutive windows"))


def render_rule_variants(results: dict) -> list[str]:
    """Report section with the thresholds of every rule variant and how they were chosen."""
    cfg = results["config"]
    variants = [m for m in RULE_METHODS if m in results["metrics"]]
    lines = ["", "## Rule variants", "",
             "| Setting | " + " | ".join(METHOD_LABELS[v] for v in variants) + " |",
             "|---|" + "---:|" * len(variants)]
    for key, label in RULE_THRESHOLD_FIELDS:
        digits = 2 if key == "gateway_5xx_ratio" else 0
        lines.append(f"| {label} | " + " | ".join(_fmt(float(cfg[v][key]), digits) for v in variants) + " |")
    tuning = results.get("rule_tuning")
    if tuning:
        grid = "; ".join(f"`{k}` ∈ {{{', '.join(str(x) for x in v)}}}" for k, v in tuning["grid"].items())
        lines += ["",
                  f"Tuned on the {tuning['train_windows']} **training windows only** ({tuning['candidates']} "
                  f"combinations: {grid}). Selection: highest training F1, ties broken by fewer false-alarm "
                  f"episodes, then by grid order. Chosen setting: training F1 {tuning['train_f1']:.3f}, "
                  f"{tuning['train_false_alarm_episodes']} false-alarm episodes "
                  f"({tuning['tied_best']} combination(s) tied on both). "
                  "`Rule-based (tuned, k=1)` uses the same thresholds with `consecutive = 1`, "
                  "so it alerts on a single window like the ML models.",
                  ""]
        edges = tuning.get("grid_edge") or []
        if edges:
            lines.append("**Grid-edge check:** the chosen value is on the edge of the grid, and the next value "
                         "inward is worse on the training windows, so a better setting may lie outside the grid. "
                         "Consider extending the grid in this direction:")
            lines.append("")
            for e in edges:
                lines.append(f"- `{e['parameter']}` = {e['value']:g} ({e['edge']} edge): training F1 {e['f1']:.3f} "
                             f"vs {e['neighbour_f1']:.3f} at {e['neighbour']:g}; false-alarm episodes "
                             f"{e['false_alarm_episodes']} vs {e['neighbour_false_alarm_episodes']}")
        else:
            lines.append("**Grid-edge check:** passed – no tuned parameter is limited by the edge of the grid "
                         "(edges where the neighbouring value scores the same are not flagged; "
                         "`consecutive = 1` is the natural minimum).")
        lines += ["", "Best grid points on the training windows:", "",
                  "| consecutive | 5xx ratio | p95 (ms) | ERROR lines | Train F1 | False-alarm episodes | Alerting windows |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in tuning["top"]:
            lines.append(f"| {r['consecutive']:.0f} | {r['gateway_5xx_ratio']:.2f} | {r['gateway_p95_ms']:.0f} | "
                         f"{r['errors_per_service']:.0f} | {r['f1']:.3f} | {r['false_alarm_episodes']:.0f} | "
                         f"{r['alerts']:.0f} |")
    return lines


def render_within_reference(results: dict, order: list[str]) -> list[str]:
    """Cross-run report only: the test run's own (chronological split) results next to the cross-run ones."""
    ref = results.get("within_run_reference")
    if not ref:
        return []
    cross = results["dataset"]["cross"]
    lines = ["", "### Cross-run vs within-run", "",
             f"Within-run = `{cross['test_run']}` evaluated on its own (trained on its first part, tested on the "
             f"windows after {ref.get('split_at')}). Cross-run = trained on `{cross['train_run']}`. A large drop for "
             "Random Forest suggests it memorised its training run; a rise in false alarms for Isolation Forest or "
             "PCA shows environment drift between runs. The test windows differ (whole run vs last part), so "
             "compare the size of the change, not single digits.", "",
             "| Method | F1 within | F1 cross | Δ F1 | False alarms / h within | False alarms / h cross | "
             "Detection rate within | Detection rate cross |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in order:
        w = ref["methods"].get(m)
        if not w:
            continue
        c, d = results["metrics"][m], results["detection_delay"][m]
        delta = c["f1"] - w["f1"] if w.get("f1") is not None else None
        lines.append(f"| {METHOD_LABELS[m]} | {_fmt(w.get('f1'))} | {_fmt(c['f1'])} | "
                     f"{'–' if delta is None else f'{delta:+.3f}'} | {_fmt(w.get('false_alarms_per_hour'), 2)} | "
                     f"{_fmt(c['false_alarms_per_hour'], 2)} | {_fmt(w.get('detection_rate'), 2)} | "
                     f"{_fmt(d['detection_rate'], 2)} |")
    return lines


def render_report(results: dict, delays: pd.DataFrame) -> str:
    ds = results["dataset"]
    cfg = results["config"]
    order = _methods(results)
    cross = ds.get("cross")
    if cross:
        dataset_lines = [
            "# Cross-run evaluation report",
            "",
            f"Every method is trained on the **whole** run `{cross['train_run']}` and tested on the **whole** run "
            f"`{cross['test_run']}`. Nothing of the test run is used for fitting (templates, rule tuning, models).",
            "",
            "## Dataset",
            "",
            f"- Training run `{cross['train_run']}`: {cross['train_period'][0]} → {cross['train_period'][1]}, "
            f"{ds['windows_train']} windows ({ds['anomalous_windows_train']} anomalous), {ds['faults_train']} faults",
            f"- Test run `{cross['test_run']}`: {cross['test_period'][0]} → {cross['test_period'][1]}, "
            f"{ds['windows_test']} windows ({ds['anomalous_windows_test']} anomalous), {ds['faults_test']} faults",
            f"- Log records: {ds['log_records']:,}; known normal templates (from the training run): "
            f"{ds['known_templates']}; {cfg['window_s']:.0f} s window, {cfg['step_s']:.0f} s step, "
            f"label grace period {cfg['grace_s']:.0f} s",
        ]
    else:
        dataset_lines = [
            "# Anomaly detection evaluation report",
            "",
            "## Dataset",
            "",
            f"- Period: {ds['period_start']} → {ds['period_end']} (train/test split at {ds['split_at']})",
            f"- Log records: {ds['log_records']:,}; known normal templates: {ds['known_templates']}",
            f"- Windows: {ds['windows_total']} ({cfg['window_s']:.0f} s window, {cfg['step_s']:.0f} s step); "
            f"train {ds['windows_train']} ({ds['anomalous_windows_train']} anomalous), "
            f"test {ds['windows_test']} ({ds['anomalous_windows_test']} anomalous), "
            f"{ds['windows_dropped_at_split']} dropped at the split",
            f"- Faults: {ds['faults_total']} injected, {ds['faults_test']} in the test period; "
            f"window label grace period {cfg['grace_s']:.0f} s",
        ]
    lines = [
        *dataset_lines,
        "",
        "## Q1 / Q2 – Detection quality on the test windows",
        "",
        "| Method | Precision | Recall | F1 | ROC-AUC | FPR | False-alarm episodes | False alarms / h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in order:
        r = results["metrics"][m]
        lines.append(f"| {METHOD_LABELS[m]} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | "
                     f"{_fmt(r['roc_auc'])} | {_fmt(r['false_positive_rate'])} | {r['false_alarm_episodes']} | "
                     f"{_fmt(r['false_alarms_per_hour'], 2)} |")
    lines += render_within_reference(results, order)
    lines += [
        "",
        "Confusion counts (windows):",
        "",
        "| Method | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in order:
        r = results["metrics"][m]
        lines.append(f"| {METHOD_LABELS[m]} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |")

    lines += [
        "",
        "## Q3 – Detection delay (end of first alerting window − fault start)",
        "",
        "| Method | Detected | Detection rate | Mean delay | Median delay | Max delay |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in order:
        d = results["detection_delay"][m]
        lines.append(f"| {METHOD_LABELS[m]} | {d['detected']}/{d['faults']} | {_fmt(d['detection_rate'], 2)} | "
                     f"{_fmt(d['mean_delay_s'], 1, ' s')} | {_fmt(d['median_delay_s'], 1, ' s')} | "
                     f"{_fmt(d['max_delay_s'], 1, ' s')} |")
    lines += [
        "",
        "### ML compared with the rules, per fault",
        "",
        "| Method | Earlier | Same | Later or missed | Both missed | Only ML detected | Mean time saved | Median time saved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, c in results["comparison_vs_rules"].items():
        lines.append(f"| {METHOD_LABELS[m]} | {c['earlier']} | {c['same']} | {c['later_or_missed']} | "
                     f"{c['both_missed']} | {c['detected_only_by_ml']} | {_fmt(c['mean_time_saved_s'], 1, ' s')} | "
                     f"{_fmt(c['median_time_saved_s'], 1, ' s')} |")
    lines.append("")
    lines.append("Time saved = rule delay − ML delay, over faults detected by both (positive = ML earlier).")
    variants = results.get("comparison_vs_rule_variants") or {}
    if len(variants) > 1:
        lines += [
            "",
            "### ML compared with each rule variant, per fault",
            "",
            "The tuned variants are the fair baseline: their thresholds were chosen on the training "
            "windows, and the k=1 variant has no built-in `for:` delay. Base conclusions on them.",
            "",
            "| Method | Rule variant | Earlier | Same | Later or missed | Both missed | Only ML detected | "
            "Mean time saved | Median time saved |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r, per_ml in variants.items():
            for m, c in per_ml.items():
                lines.append(f"| {METHOD_LABELS[m]} | {METHOD_LABELS[r]} | {c['earlier']} | {c['same']} | "
                             f"{c['later_or_missed']} | {c['both_missed']} | {c['detected_only_by_ml']} | "
                             f"{_fmt(c['mean_time_saved_s'], 1, ' s')} | {_fmt(c['median_time_saved_s'], 1, ' s')} |")

    for title, key in (("By fault type", "by_fault_type"), ("By target service", "by_service")):
        if not results[key]:
            continue
        lines += ["", f"### {title}", "",
                  "| Group | Faults | " + " | ".join(f"{METHOD_LABELS[m]} detected / mean delay" for m in order) + " |",
                  "|---|---:|" + "---:|" * len(order)]
        for group, stats in results[key].items():
            cells = []
            for m in order:
                s = stats[m]
                cells.append(f"{s['detected']}/{s['faults']} / {_fmt(s['mean_delay_s'], 1, ' s')}")
            lines.append(f"| {group} | {stats['faults']} | " + " | ".join(cells) + " |")

    if any(m in results["metrics"] for m in RULE_METHODS[1:]):
        lines += render_rule_variants(results)

    if results["feature_importance"]:
        lines += ["", "## Random Forest feature importance (top 20)", "", "| Feature | Importance |", "|---|---:|"]
        for item in results["feature_importance"]:
            lines.append(f"| `{item['feature']}` | {item['importance']:.4f} |")
    if results.get("logistic_regression_coefficients"):
        lines += ["", "## Logistic Regression coefficients (top 20 by size)", "",
                  "Coefficients on standardised log1p features; positive = pushes towards anomalous.", "",
                  "| Feature | Coefficient |", "|---|---:|"]
        for item in results["logistic_regression_coefficients"]:
            lines.append(f"| `{item['feature']}` | {item['coefficient']:+.3f} |")

    if len(delays):
        lines += ["", "## Per-fault detection delays", "",
                  "| Fault | Service | Type | Duration | " + " | ".join(METHOD_LABELS[m] for m in order) + " |",
                  "|---|---|---|---:|" + "---:|" * len(order)]
        for _, r in delays.iterrows():
            cells = [(_fmt(r[f"{m}_delay_s"], 0, " s") if r[f"{m}_detected"] else "missed") for m in order]
            lines.append(f"| {r['fault_id']} | {r['service']} | {r['type']} | {r['duration_s']:.0f} s | "
                         + " | ".join(cells) + " |")

    lines += ["", "## Figures", "",
              "- `timeline.png` – test period: gateway latency, error lines, faults and alerts per method",
              "- `metrics.png` – precision, recall and F1 per method",
              "- `detection_delay.png` – distribution of per-fault detection delays", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.titlecolor": TEXT,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def plot_timeline(test: pd.DataFrame, labels: pd.DataFrame, preds: dict[str, np.ndarray],
                  t_split: float, grace: float, path: Path) -> None:
    plt = _plt()
    t0 = float(test["window_start"].min())
    x = (test["window_end"].to_numpy() - t0) / 60.0
    n_methods = sum(1 for m in METHOD_ORDER if m in preds)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.2 + 0.35 * n_methods), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 0.35 + 0.42 * n_methods]})
    faults = [(f.start.timestamp(), f.end.timestamp(), f.type, f.service)
              for f in labels.itertuples(index=False) if f.end.timestamp() >= t0]

    def shade(ax):
        for s, e, *_ in faults:
            ax.axvspan((s - t0) / 60.0, (e - t0) / 60.0, color=FAULT_SHADE, alpha=0.6, linewidth=0)

    ax = axes[0]
    _style_axes(ax)
    shade(ax)
    p95 = test[feature_name(get_profile().entry_service, "latency_p95")].to_numpy() / 1000.0
    ax.plot(x, p95, color=TEXT_SECONDARY, linewidth=1.5)
    ax.set_ylim(0, max(float(np.nanmax(p95)) if p95.size else 0.0, 0.1) * 1.35)
    ax.set_ylabel("p95 latency (s)")
    ax.set_title(f"{get_profile().entry_service} p95 latency per window (shaded = injected fault)")
    for s, e, ftype, svc in faults:
        ax.annotate(f"{ftype}\n{svc.replace('-service', '')}", xy=(((s + e) / 2 - t0) / 60.0, 1.0),
                    xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7,
                    color=TEXT_SECONDARY)

    ax = axes[1]
    _style_axes(ax)
    shade(ax)
    errors = sum(test[feature_name(s, "error_count")].to_numpy() for s in get_profile().services)
    ax.plot(x, errors, color=TEXT_SECONDARY, linewidth=1.5)
    ax.set_ylabel("ERROR lines")
    ax.set_title("ERROR log lines per window, all services")

    ax = axes[2]
    _style_axes(ax)
    ax.grid(False)
    shade(ax)
    width = (test["window_end"].iloc[0] - test["window_start"].iloc[0]) / 60.0 if len(test) else 0.5
    step_min = (test["window_start"].diff().median() / 60.0) if len(test) > 1 else width
    order = [m for m in METHOD_ORDER if m in preds]
    for row, m in enumerate(order):
        idx = np.flatnonzero(preds[m] == 1)
        bars = [(x[i] - step_min, step_min) for i in idx]
        ax.broken_barh(bars, (row - 0.35, 0.7), facecolors=COLORS[m], edgecolor=SURFACE, linewidth=0.3)
    ax.set_yticks(range(len(order)), [METHOD_LABELS[m] for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_title("Alerts (window end time)")
    ax.set_xlabel("Minutes since start of test period")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics(metrics: dict, path: Path) -> None:
    plt = _plt()
    names = ["precision", "recall", "f1"]
    order = [m for m in METHOD_ORDER if m in metrics]
    fig, ax = plt.subplots(figsize=(4 + 1.2 * len(order), 4.4))
    _style_axes(ax)
    n = len(order)
    width = 0.8 / n
    xs = np.arange(len(names))
    for i, m in enumerate(order):
        values = [metrics[m][k] for k in names]
        pos = xs - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, values, width=width * 0.92, color=COLORS[m], label=METHOD_LABELS[m])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8 if n <= 3 else 6.5, color=TEXT_SECONDARY)
    ax.set_xticks(xs, ["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (test windows)")
    ax.set_title("Detection quality per method")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(n, 3), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delays(delays: pd.DataFrame, methods: list[str], path: Path) -> None:
    plt = _plt()
    order = [m for m in METHOD_ORDER if m in methods]
    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.85 * len(order)))
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    rng = np.random.default_rng(0)
    labels = []
    for row, m in enumerate(order):
        if len(delays):
            d = delays.loc[delays[f"{m}_detected"], f"{m}_delay_s"].to_numpy(dtype=float)
            total = len(delays)
        else:
            d, total = np.empty(0), 0
        if d.size:
            ax.boxplot(d, positions=[row], vert=False, widths=0.5, showfliers=False,
                       medianprops={"color": TEXT, "linewidth": 1.5},
                       boxprops={"color": TEXT_SECONDARY}, whiskerprops={"color": TEXT_SECONDARY},
                       capprops={"color": TEXT_SECONDARY})
            ax.scatter(d, row + rng.uniform(-0.15, 0.15, d.size), s=36, color=COLORS[m],
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
        labels.append(f"{METHOD_LABELS[m]}\n{d.size}/{total} detected")
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Detection delay (s) – end of first alerting window minus fault start")
    ax.set_title("Detection delay per injected fault (test period)")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
