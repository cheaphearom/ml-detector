"""Aggregate several evaluated experiment runs.

Each run folder (e.g. ``results/run_3/``) holds ``labels.csv``, ``experiment.json`` and the
output of ``detector.cli evaluate`` in ``detector/`` (``results.json``,
``detection_delays.csv``).

Outputs in ``out_dir``:
  summary_metrics.csv / .md   per method: mean ± standard deviation over runs of
                              Precision, Recall, F1, ROC-AUC, FPR, false alarms per hour
  pooled_delays.csv           one row per test-period fault x method, from every run
  summary_q3.md               pooled detection rate and delays, earlier / same / later
                              counts against each rule variant, by fault type and severity
  stats.md, stats_tests.csv    McNemar / Wilcoxon tests (Holm-corrected) and plain-English
                              lines per hypothesis, see :mod:`detector.stats`
  ci_f1.csv, ci_delay.csv     95 % cluster-bootstrap confidence intervals
  f1_by_method.png, pooled_delays.png, detection_rate_by_type.png   (300 dpi)
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import COLORS, GRID, SURFACE, TEXT, TEXT_SECONDARY, _fmt, _plt, _style_axes, compare_with_rules
from . import stats as st
from .models import METHOD_LABELS, METHOD_ORDER, ML_METHODS, RULE_METHODS

METRICS = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"),
           ("false_positive_rate", "FPR"), ("false_alarms_per_hour", "False alarms / h"))
FAULT_COLUMNS = ("run_id", "fault_id", "type", "severity", "service", "start", "end", "duration_s")
SEVERITY_ORDER = ("low", "medium", "high")
DPI = 300


@dataclass
class Run:
    run_id: str
    path: Path
    results: dict
    delays: pd.DataFrame          # wide, as written by ``evaluate`` (one row per fault)
    windows: pd.DataFrame | None = None   # window_predictions.csv (older runs may lack it)

    @property
    def methods(self) -> list[str]:
        return [m for m in METHOD_ORDER if m in self.results["metrics"]]


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------
def expand_runs(patterns: list[str]) -> list[Path]:
    """Run folders from paths or glob patterns (the shell may or may not have expanded them)."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        paths += [Path(m) for m in matches if Path(m).is_dir()]
    unique = list(dict.fromkeys(p.resolve() for p in paths))
    return sorted(unique, key=lambda p: _natural_key(p.name))


def _natural_key(name: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def load_run(path: str | os.PathLike, detector_dir: str = "detector") -> Run:
    """Loads one evaluated run; raises ``ValueError`` with the missing file if it is incomplete."""
    path = Path(path)
    res_path = path / detector_dir / "results.json"
    delays_path = path / detector_dir / "detection_delays.csv"
    for f in (res_path, delays_path):
        if not f.exists():
            raise ValueError(f"{f} is missing - run `python -m detector.cli evaluate --out {path / detector_dir}` "
                             f"for this run first")
    results = json.loads(res_path.read_text(encoding="utf-8"))
    delays = pd.read_csv(delays_path, dtype={"fault_id": str})
    win_path = path / detector_dir / "window_predictions.csv"
    windows = pd.read_csv(win_path, dtype={"fault_id": str, "cluster": str}) if win_path.exists() else None
    return Run(run_id=path.name, path=path, results=results, delays=delays, windows=windows)


def load_runs(patterns: list[str], detector_dir: str = "detector") -> list[Run]:
    paths = expand_runs(patterns)
    if not paths:
        raise ValueError(f"no run folders match {' '.join(patterns)}")
    runs = [load_run(p, detector_dir) for p in paths]
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"run folder names must be unique, got {ids}")
    check_same_settings(runs)
    return runs


COMPARABLE_SETTINGS = ("window_s", "step_s", "train_frac", "grace_s")


def check_same_settings(runs: list[Run]) -> None:
    """Refuses to mix runs evaluated with different window, step, train fraction or grace period."""
    seen: dict[str, set] = {}
    for r in runs:
        cfg = r.results.get("config", {})
        for key in COMPARABLE_SETTINGS:
            seen.setdefault(key, set()).add(cfg.get(key))
    differing = {k: sorted(map(str, v)) for k, v in seen.items() if len(v) > 1}
    if differing:
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in differing.items())
        raise ValueError(f"the runs were evaluated with different settings ({detail}); "
                         f"re-run `evaluate` with the same options for every run")


def severity_of(params) -> str:
    """Severity from the ``params`` JSON of a label (``{"severity": "low", ...}``)."""
    if not isinstance(params, str) or not params.strip():
        return "unknown"
    try:
        value = json.loads(params).get("severity")
    except (ValueError, AttributeError):
        return "unknown"
    return str(value) if value else "unknown"


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------
def methods_of(runs: list[Run]) -> list[str]:
    present = set().union(*(r.methods for r in runs)) if runs else set()
    return [m for m in METHOD_ORDER if m in present]


def per_run_metrics(runs: list[Run]) -> pd.DataFrame:
    """Long table: one row per run x method with every metric."""
    rows = []
    for r in runs:
        for m in r.methods:
            met = r.results["metrics"][m]
            rows.append({"run_id": r.run_id, "method": m,
                         **{k: (np.nan if met.get(k) is None else float(met[k])) for k, _ in METRICS}})
    return pd.DataFrame(rows, columns=["run_id", "method", *[k for k, _ in METRICS]])


def summary_metrics(runs: list[Run]) -> pd.DataFrame:
    """Per method: mean and sample standard deviation (ddof = 1) over runs of every metric."""
    long = per_run_metrics(runs)
    rows = []
    for m in methods_of(runs):
        sub = long[long["method"] == m]
        row = {"method": m, "label": METHOD_LABELS[m], "runs": int(len(sub))}
        for k, _ in METRICS:
            v = sub[k].dropna()
            row[f"{k}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def render_summary_metrics(summary: pd.DataFrame, n_runs: int) -> str:
    lines = [f"# Detection quality over {n_runs} runs", "",
             "Mean ± sample standard deviation over runs (test windows of each run).", "",
             "| Method | Runs | " + " | ".join(label for _, label in METRICS) + " |",
             "|---|---:|" + "---:|" * len(METRICS)]
    for _, r in summary.iterrows():
        cells = []
        for k, _ in METRICS:
            digits = 2 if k == "false_alarms_per_hour" else 3
            mean, std = r[f"{k}_mean"], r[f"{k}_std"]
            cells.append("–" if pd.isna(mean) else
                         f"{mean:.{digits}f}" + ("" if pd.isna(std) else f" ± {std:.{digits}f}"))
        lines.append(f"| {r['label']} | {r['runs']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def pooled_wide(runs: list[Run]) -> pd.DataFrame:
    """Every run's ``detection_delays.csv`` stacked, with ``run_id`` and ``severity`` added."""
    frames = []
    for r in runs:
        d = r.delays.copy()
        d.insert(0, "run_id", r.run_id)
        d.insert(3, "severity", d["params"].map(severity_of) if "params" in d else "unknown")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=list(FAULT_COLUMNS))
    wide = pd.concat(frames, ignore_index=True, sort=False)
    for m in methods_of(runs):
        col = f"{m}_detected"
        if col in wide:
            # a method that was not evaluated in a run stays missing (NaN), never "not detected"
            wide[col] = wide[col].map(lambda v: v if pd.isna(v) else str(v).lower() == "true").astype(object)
    return wide


def pooled_delays(runs: list[Run]) -> pd.DataFrame:
    """Long table: one row per test-period fault x method (only methods evaluated in that run)."""
    wide = pooled_wide(runs)
    rows = []
    for _, f in wide.iterrows():
        base = {c: f.get(c) for c in FAULT_COLUMNS}
        for m in methods_of(runs):
            det = f.get(f"{m}_detected")
            if det is None or (isinstance(det, float) and math.isnan(det)):
                continue
            delay = f.get(f"{m}_delay_s")
            rows.append({**base, "method": m, "detected": bool(det),
                         "delay_s": float(delay) if bool(det) and pd.notna(delay) else np.nan})
    return pd.DataFrame(rows, columns=[*FAULT_COLUMNS, "method", "detected", "delay_s"])


def detection_summary(long: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Per method: faults, detected, detection rate, mean / median delay of detected faults."""
    rows = []
    for m in methods:
        sub = long[long["method"] == m]
        d = sub.loc[sub["detected"], "delay_s"]
        rows.append({"method": m, "faults": int(len(sub)), "detected": int(sub["detected"].sum()),
                     "detection_rate": float(sub["detected"].mean()) if len(sub) else np.nan,
                     "mean_delay_s": float(d.mean()) if len(d) else np.nan,
                     "median_delay_s": float(d.median()) if len(d) else np.nan})
    return pd.DataFrame(rows)


def pooled_comparisons(wide: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Earlier / same / later counts of every ML model against every rule variant, pooled over runs.

    Only faults from runs in which both methods were evaluated take part.
    """
    rows = []
    for rule in [m for m in methods if m in RULE_METHODS]:
        for ml in [m for m in methods if m in ML_METHODS]:
            cols = [f"{rule}_detected", f"{ml}_detected"]
            if not all(c in wide for c in cols):
                continue
            both = wide.dropna(subset=cols)
            c = compare_with_rules(both, ml, rule=rule)
            rows.append({"method": ml, "rule": rule, "faults": int(len(both)), **c})
    return pd.DataFrame(rows)


def grouped(wide: pd.DataFrame, long: pd.DataFrame, by: str, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detection summary and rule comparisons per value of ``by`` (``type`` or ``severity``)."""
    det, comp = [], []
    for key in _group_order(long[by].dropna().unique(), by):
        d = detection_summary(long[long[by] == key], methods)
        d.insert(0, by, key)
        det.append(d)
        c = pooled_comparisons(wide[wide[by] == key], methods)
        if len(c):
            c.insert(0, by, key)
            comp.append(c)
    return (pd.concat(det, ignore_index=True) if det else pd.DataFrame(),
            pd.concat(comp, ignore_index=True) if comp else pd.DataFrame())


def _group_order(values, by: str) -> list:
    values = [str(v) for v in values]
    if by == "severity":
        known = [s for s in SEVERITY_ORDER if s in values]
        return known + sorted(v for v in values if v not in SEVERITY_ORDER)
    return sorted(values)


def render_q3(runs: list[Run], wide: pd.DataFrame, long: pd.DataFrame) -> str:
    methods = methods_of(runs)
    overall = detection_summary(long, methods)
    comp = pooled_comparisons(wide, methods)
    n_faults = int(len(wide))
    lines = [f"# Q3 – Detection delay pooled over {len(runs)} runs", "",
             f"Runs: {', '.join(r.run_id for r in runs)}. Test-period faults: {n_faults}. "
             "Delay = end of the first alerting window − fault start; mean and median over detected faults.", "",
             "| Method | Detected | Detection rate | Mean delay | Median delay |",
             "|---|---:|---:|---:|---:|"]
    for _, r in overall.iterrows():
        lines.append(f"| {METHOD_LABELS[r['method']]} | {r['detected']}/{r['faults']} | "
                     f"{_fmt(r['detection_rate'], 2)} | {_fmt(r['mean_delay_s'], 1, ' s')} | "
                     f"{_fmt(r['median_delay_s'], 1, ' s')} |")
    lines += ["", "## ML compared with each rule variant, per fault", "",
              "Earlier = ML alerted first, or only ML detected the fault. Later or missed = the rule alerted "
              "first, or only the rule detected it. Time saved = rule delay − ML delay over faults both "
              "detected. The tuned variants are the fair baseline for the thesis conclusion.", ""]
    lines += _comparison_table(comp)
    for by, title in (("type", "fault type"), ("severity", "severity")):
        det, gcomp = grouped(wide, long, by, methods)
        if det.empty:
            continue
        lines += ["", f"## By {title}", "", "Detection rate (detected / faults) and mean delay per method.", "",
                  f"| {title.capitalize()} | Faults | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |",
                  "|---|---:|" + "---:|" * len(methods)]
        for key, g in det.groupby(by, sort=False):
            g = g.set_index("method")
            cells = [f"{g.at[m, 'detected']}/{g.at[m, 'faults']} · {_fmt(g.at[m, 'mean_delay_s'], 1, ' s')}"
                     if m in g.index and g.at[m, 'faults'] else "–" for m in methods]
            lines.append(f"| {key} | {int(g['faults'].max())} | " + " | ".join(cells) + " |")
        if not gcomp.empty:
            lines += ["", f"### ML compared with each rule variant, by {title}", ""]
            lines += _comparison_table(gcomp, group=by)
    return "\n".join(lines) + "\n"


def _comparison_table(comp: pd.DataFrame, group: str | None = None) -> list[str]:
    if comp.empty:
        return ["No ML model and rule variant were evaluated in the same runs."]
    head = f"| {group.capitalize()} " if group else ""
    lines = [head + "| Method | Rule variant | Faults | Earlier | Same | Later or missed | Both missed | "
                    "Mean time saved | Median time saved |",
             ("|---" if group else "") + "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, c in comp.iterrows():
        g = f"| {c[group]} " if group else ""
        lines.append(g + f"| {METHOD_LABELS[c['method']]} | {METHOD_LABELS[c['rule']]} | {c['faults']} | "
                         f"{c['earlier']} | {c['same']} | {c['later_or_missed']} | {c['both_missed']} | "
                         f"{_fmt(c['mean_time_saved_s'], 1, ' s')} | {_fmt(c['median_time_saved_s'], 1, ' s')} |")
    return lines


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def plot_f1(runs: list[Run], summary: pd.DataFrame, path: Path) -> None:
    """Mean F1 per method with ±1 SD error bars; each run as a dot."""
    plt = _plt()
    long = per_run_metrics(runs)
    methods = list(summary["method"])
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(methods), 4.6))
    _style_axes(ax)
    xs = np.arange(len(methods))
    means = summary["f1_mean"].to_numpy(dtype=float)
    stds = np.nan_to_num(summary["f1_std"].to_numpy(dtype=float))
    ax.bar(xs, means, width=0.62, color=[COLORS[m] for m in methods], zorder=2)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=4, zorder=3)
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = long.loc[long["method"] == m, "f1"].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=14, color=SURFACE, edgecolors=TEXT,
                   linewidths=0.8, zorder=4)
        ax.text(i, min(means[i] + stds[i], 1.0) + 0.03, f"{means[i]:.2f}", ha="center", va="bottom",
                fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(xs, [METHOD_LABELS[m].replace(" (", "\n(") for m in methods], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 (test windows)")
    ax.set_title(f"F1 per method over {len(runs)} runs (bar = mean, line = ±1 SD, dots = runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_pooled_delays(long: pd.DataFrame, methods: list[str], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 1.6 + 0.8 * len(methods)))
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    rng = np.random.default_rng(0)
    labels = []
    for row, m in enumerate(methods):
        sub = long[long["method"] == m]
        d = sub.loc[sub["detected"], "delay_s"].dropna().to_numpy(dtype=float)
        if d.size:
            ax.boxplot(d, positions=[row], vert=False, widths=0.55, showfliers=False,
                       medianprops={"color": TEXT, "linewidth": 1.5}, boxprops={"color": TEXT_SECONDARY},
                       whiskerprops={"color": TEXT_SECONDARY}, capprops={"color": TEXT_SECONDARY})
            ax.scatter(d, row + rng.uniform(-0.18, 0.18, d.size), s=14, color=COLORS[m], alpha=0.75,
                       edgecolors="none", zorder=3)
        labels.append(f"{METHOD_LABELS[m]}\n{d.size}/{len(sub)} detected")
    ax.set_yticks(range(len(methods)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Detection delay (s) – end of first alerting window minus fault start")
    ax.set_title("Detection delay per fault, pooled over runs")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_rate_by_type(long: pd.DataFrame, methods: list[str], path: Path) -> None:
    plt = _plt()
    types = _group_order(long["type"].dropna().unique(), "type")
    fig, ax = plt.subplots(figsize=(2.5 + 2.2 * max(len(types), 1), 4.6))
    _style_axes(ax)
    width = 0.84 / max(len(methods), 1)
    xs = np.arange(len(types))
    for i, m in enumerate(methods):
        rates = []
        for t in types:
            sub = long[(long["method"] == m) & (long["type"] == t)]
            rates.append(sub["detected"].mean() if len(sub) else np.nan)
        pos = xs - 0.42 + width * (i + 0.5)
        ax.bar(pos, rates, width=width * 0.9, color=COLORS[m], label=METHOD_LABELS[m], zorder=2)
    n_per_type = [long.loc[long["type"] == t, ["run_id", "fault_id"]].drop_duplicates().shape[0] for t in types]
    ax.set_xticks(xs, [f"{t}\n({n} faults)" for t, n in zip(types, n_per_type)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate (pooled faults)")
    ax.set_title("Detection rate by fault type and method")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(len(methods), 4), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------
def statistics(runs: list[Run], wide: pd.DataFrame, n_boot: int = st.N_BOOT, seed: int = st.SEED) -> dict:
    """Confidence intervals and paired tests (see :mod:`detector.stats`)."""
    methods = methods_of(runs)
    if all(r.windows is not None for r in runs):
        windows = pd.concat([r.windows.assign(run_id=r.run_id) for r in runs], ignore_index=True, sort=False)
        f1_ci = st.f1_cluster_ci(windows, methods, n_boot, seed)
    else:
        f1_ci = st.f1_run_ci(per_run_metrics(runs), methods, n_boot, seed)
    return {"f1_ci": f1_ci, "delay_ci": st.delay_ci(wide, methods, n_boot, seed),
            "tests": st.paired_tests(wide, methods)}


def aggregate(run_patterns: list[str], out_dir: str | os.PathLike, make_figures: bool = True,
              detector_dir: str = "detector", n_boot: int = st.N_BOOT, seed: int = st.SEED) -> dict:
    """Loads every run, writes the summary tables and figures, and returns the main tables."""
    runs = load_runs(run_patterns, detector_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = methods_of(runs)

    summary = summary_metrics(runs)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    (out / "summary_metrics.md").write_text(render_summary_metrics(summary, len(runs)), encoding="utf-8")

    wide = pooled_wide(runs)
    long = pooled_delays(runs)
    long.to_csv(out / "pooled_delays.csv", index=False)
    (out / "summary_q3.md").write_text(render_q3(runs, wide, long), encoding="utf-8")

    stat = statistics(runs, wide, n_boot, seed)
    stat["f1_ci"].to_csv(out / "ci_f1.csv", index=False)
    stat["delay_ci"].to_csv(out / "ci_delay.csv", index=False)
    stat["tests"].to_csv(out / "stats_tests.csv", index=False)
    (out / "stats.md").write_text(st.render_stats(stat["f1_ci"], stat["delay_ci"], stat["tests"], len(runs),
                                                  n_boot, seed), encoding="utf-8")

    if make_figures:
        plot_f1(runs, summary, out / "f1_by_method.png")
        plot_pooled_delays(long, methods, out / "pooled_delays.png")
        plot_rate_by_type(long, methods, out / "detection_rate_by_type.png")
    return {"runs": [r.run_id for r in runs], "methods": methods, "summary": summary,
            "pooled_delays": long, "comparisons": pooled_comparisons(wide, methods), **stat,
            "outputs": sorted(p.name for p in out.iterdir() if p.is_file())}
