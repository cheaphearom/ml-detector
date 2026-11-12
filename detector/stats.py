"""Statistical tests for the aggregated runs (used by ``detector.cli aggregate``).

* 95 % cluster-bootstrap confidence intervals (2,000 resamples, fixed seed):
    - F1 per method: test windows are grouped into clusters (one fault plus the normal gap after
      it, see :func:`detector.evaluate.window_clusters`); clusters are resampled within each run
      and F1 is computed from the summed TP / FP / FN.
    - mean detection delay per method: faults are resampled within each run; the statistic is
      the mean delay of the detected faults in the resample.
* McNemar's exact test on per-fault detection, each ML model vs each tuned rule variant.
* Wilcoxon signed-rank test on paired delays (faults detected by both), each ML model vs each
  tuned rule variant, with n, median difference and effect size r = |Z| / sqrt(n).
* Holm correction over every test that was run. alpha = 0.05.

No test is run on fewer than ``MIN_PAIRS`` informative pairs; the output then says so.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

from .models import METHOD_LABELS, ML_METHODS

ALPHA = 0.05
N_BOOT = 2000
SEED = 42
MIN_PAIRS = 6
TUNED_RULES = ("rules_tuned", "rules_tuned_k1")


# ---------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------
def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values; ``None`` (test not run) stays ``None``."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    m = len(idx)
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def stratified_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """``n_boot`` x ``len(strata)`` resampling indices, sampling with replacement inside each stratum."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    idx = np.empty((n_boot, len(strata)), dtype=np.int64)
    col = 0
    for s in pd.unique(strata):
        members = np.flatnonzero(strata == s)
        idx[:, col:col + members.size] = rng.choice(members, size=(n_boot, members.size), replace=True)
        col += members.size
    return idx


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (math.nan, math.nan)
    lo, hi = np.percentile(samples, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return (float(lo), float(hi))


def _f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, 0.0)


# ---------------------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------------------
def f1_cluster_ci(windows: pd.DataFrame, methods: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Pooled F1 per method with a cluster-bootstrap 95 % CI.

    ``windows`` holds ``run_id``, ``cluster``, ``label`` and ``<method>_pred`` for the test windows
    of every run. All methods share the same resamples.
    """
    keys = windows["run_id"].astype(str) + "|" + windows["cluster"].astype(str)
    codes, uniq = pd.factorize(keys)
    cluster_run = pd.Series(windows["run_id"].to_numpy()).groupby(codes).first().to_numpy()
    idx = stratified_indices(cluster_run, n_boot, seed)
    y = windows["label"].to_numpy(dtype=int)
    rows = []
    for m in methods:
        col = f"{m}_pred"
        if col not in windows:
            continue
        sub = windows[col].notna().to_numpy()
        pred = windows[col].fillna(0).to_numpy(dtype=int)
        counts = np.zeros((len(uniq), 3))
        np.add.at(counts, codes[sub], np.column_stack([(pred == 1) & (y == 1), (pred == 1) & (y == 0),
                                                       (pred == 0) & (y == 1)])[sub].astype(float))
        boot = counts[idx].sum(axis=1)
        tp, fp, fn = counts.sum(axis=0)
        lo, hi = percentile_ci(_f1(boot[:, 0], boot[:, 1], boot[:, 2]))
        rows.append({"method": m, "f1_pooled": float(_f1(tp, fp, fn)), "f1_ci_low": lo, "f1_ci_high": hi,
                     "windows": int(sub.sum()), "clusters": int(len(np.unique(codes[sub]))),
                     "ci_method": f"cluster bootstrap by fault, {n_boot} resamples"})
    return pd.DataFrame(rows)


def f1_run_ci(per_run: pd.DataFrame, methods: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Fallback when window predictions are missing: resample runs, statistic = mean F1."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in methods:
        v = per_run.loc[per_run["method"] == m, "f1"].dropna().to_numpy(dtype=float)
        if v.size < 2:
            lo = hi = math.nan
        else:
            lo, hi = percentile_ci(rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1))
        rows.append({"method": m, "f1_pooled": float(v.mean()) if v.size else math.nan, "f1_ci_low": lo,
                     "f1_ci_high": hi, "windows": None, "clusters": int(v.size),
                     "ci_method": f"bootstrap over runs (mean F1), {n_boot} resamples"})
    return pd.DataFrame(rows)


def delay_ci(wide: pd.DataFrame, methods: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Mean delay of detected faults per method with a 95 % CI from resampling faults within runs."""
    if wide.empty:
        return pd.DataFrame(columns=["method", "faults", "detected", "mean_delay_s", "delay_ci_low", "delay_ci_high"])
    idx = stratified_indices(wide["run_id"].to_numpy(), n_boot, seed)
    rows = []
    for m in methods:
        det_col, d_col = f"{m}_detected", f"{m}_delay_s"
        if det_col not in wide:
            continue
        evaluated = wide[det_col].notna().to_numpy()
        det = wide[det_col].map(lambda v: bool(v) if pd.notna(v) else False).to_numpy(dtype=bool) & evaluated
        delay = np.where(det, pd.to_numeric(wide[d_col], errors="coerce").to_numpy(dtype=float), 0.0)
        sums = delay[idx].sum(axis=1)
        counts = det[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.where(counts > 0, sums / counts, np.nan)
        lo, hi = percentile_ci(boot)
        rows.append({"method": m, "faults": int(evaluated.sum()), "detected": int(det.sum()),
                     "mean_delay_s": float(delay[det].mean()) if det.any() else math.nan,
                     "delay_ci_low": lo, "delay_ci_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------------------
def mcnemar_exact(ml_detected, rule_detected, min_pairs: int = MIN_PAIRS) -> dict:
    """Exact (binomial) McNemar test on paired per-fault detection.

    Only discordant faults carry information: ``ml_only`` (ML detected, rule missed) and
    ``rule_only``. Under H0 each discordant fault is equally likely to go either way.
    """
    a = np.asarray(ml_detected, dtype=bool)
    b = np.asarray(rule_detected, dtype=bool)
    ml_only, rule_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = ml_only + rule_only
    res = {"test": "McNemar (exact)", "n_faults": int(a.size), "ml_only": ml_only, "rule_only": rule_only,
           "n": n, "p": None, "note": ""}
    if n < min_pairs:
        res["note"] = f"only {n} faults detected by exactly one of the two (need ≥ {min_pairs})"
        return res
    res["p"] = float(st.binomtest(ml_only, n, 0.5, alternative="two-sided").pvalue)
    return res


def wilcoxon_delays(ml_delay, rule_delay, min_pairs: int = MIN_PAIRS) -> dict:
    """Wilcoxon signed-rank test on paired delays of faults detected by both methods.

    Differences are ``rule − ML`` (positive = ML earlier). Zero differences (both alerted on the
    same window) are dropped, as in Wilcoxon's original test; ``n`` is the number of non-zero
    pairs. The p-value and Z come from the normal approximation with tie correction (delays are
    multiples of the 10 s step, so ties are common and the exact distribution does not apply).
    Effect size r = |Z| / sqrt(n).
    """
    ml = np.asarray(ml_delay, dtype=float)
    rule = np.asarray(rule_delay, dtype=float)
    diff = rule - ml
    nonzero = diff[diff != 0]
    res = {"test": "Wilcoxon signed-rank", "n_pairs": int(diff.size), "n": int(nonzero.size),
           "median_diff_s": float(np.median(diff)) if diff.size else None,
           "mean_diff_s": float(np.mean(diff)) if diff.size else None,
           "z": None, "r": None, "p": None, "note": ""}
    if nonzero.size < min_pairs:
        res["note"] = (f"only {nonzero.size} faults detected by both with different delays "
                       f"(need ≥ {min_pairs})")
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = st.wilcoxon(nonzero, zero_method="wilcox", correction=False, method="approx")
    z = float(out.zstatistic)
    res.update(p=float(out.pvalue), z=z, r=abs(z) / math.sqrt(nonzero.size))
    return res


def identical_rules(wide: pd.DataFrame, a: str, b: str) -> bool:
    """True if two rule variants detected the same faults with the same delays (e.g. when the
    tuning already chose ``consecutive = 1``, so ``rules_tuned_k1`` equals ``rules_tuned``)."""
    cols = [f"{a}_detected", f"{b}_detected", f"{a}_delay_s", f"{b}_delay_s"]
    if wide.empty or not all(c in wide for c in cols):
        return False
    da, db = wide[f"{a}_detected"], wide[f"{b}_detected"]
    same_det = (da.isna() == db.isna()).all() and (da.dropna().astype(bool) == db.dropna().astype(bool)).all()
    xa, xb = pd.to_numeric(wide[f"{a}_delay_s"]), pd.to_numeric(wide[f"{b}_delay_s"])
    return bool(same_det and ((xa == xb) | (xa.isna() & xb.isna())).all())


def paired_tests(wide: pd.DataFrame, methods: list[str], rules: tuple[str, ...] = TUNED_RULES,
                 alpha: float = ALPHA) -> pd.DataFrame:
    """McNemar and Wilcoxon for every ML model vs every tuned rule variant, Holm-corrected together.

    A rule variant whose per-fault results are identical to an earlier variant is not tested again
    (its tests would be exact duplicates and would only inflate the Holm correction); it is listed
    in ``tests.attrs["identical"]``.
    """
    rows = []
    rules = [r for r in rules if r in methods]
    identical = {r: first for i, r in enumerate(rules) for first in rules[:i] if identical_rules(wide, first, r)}
    for rule in [r for r in rules if r not in identical]:
        for ml in [m for m in methods if m in ML_METHODS]:
            cols = [f"{rule}_detected", f"{ml}_detected"]
            if not all(c in wide for c in cols):
                continue
            both = wide.dropna(subset=cols)
            ml_det = both[f"{ml}_detected"].astype(bool).to_numpy()
            rule_det = both[f"{rule}_detected"].astype(bool).to_numpy()
            rows.append({"method": ml, "rule": rule, "hypothesis": "detection",
                         **mcnemar_exact(ml_det, rule_det)})
            pair = ml_det & rule_det
            rows.append({"method": ml, "rule": rule, "hypothesis": "delay",
                         **wilcoxon_delays(both.loc[pair, f"{ml}_delay_s"], both.loc[pair, f"{rule}_delay_s"])})
    tests = pd.DataFrame(rows)
    tests.attrs["identical"] = identical
    if tests.empty:
        return tests
    for col in ("n_faults", "ml_only", "rule_only", "n", "n_pairs"):
        if col in tests:
            tests[col] = tests[col].astype("Int64")
    tests["p_holm"] = holm(list(tests["p"]))
    tests["reject"] = [bool(p is not None and not pd.isna(p) and p < alpha) for p in tests["p_holm"]]
    tests["alpha"] = alpha
    return tests


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------
def _p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _ci(lo, hi, digits: int = 3, unit: str = "") -> str:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return "–"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]{unit}"


def hypothesis_line(t: pd.Series) -> str:
    """One plain-English sentence for one test."""
    ml, rule = METHOD_LABELS[t["method"]], t["rule"]
    if t["hypothesis"] == "delay":
        name = "H3₀ (no difference in detection delay)"
        if pd.isna(t["p"]):
            return f"{name} not tested for {ml} vs {rule}: {t['note']}."
        md = t["median_diff_s"] or 0.0
        direction = (f"median {abs(md):.0f} s {'earlier' if md > 0 else 'later'}" if md else "median difference 0 s")
        verdict = "rejected" if t["reject"] else "not rejected"
        return (f"{name} {verdict} for {ml} vs {rule} ({direction}, p = {_p(t['p_holm'])} Holm, "
                f"n = {t['n']} faults with different delays of {t['n_pairs']} detected by both, r = {t['r']:.2f}).")
    name = "H₀ (no difference in per-fault detection)"
    if pd.isna(t["p"]):
        return f"{name} not tested for {ml} vs {rule}: {t['note']}."
    verdict = "rejected" if t["reject"] else "not rejected"
    return (f"{name} {verdict} for {ml} vs {rule} ({ml} alone detected {t['ml_only']}, {rule} alone "
            f"{t['rule_only']} of {t['n_faults']} faults, exact McNemar p = {_p(t['p_holm'])} Holm).")


def render_stats(f1_ci: pd.DataFrame, d_ci: pd.DataFrame, tests: pd.DataFrame, n_runs: int,
                 n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA) -> str:
    lines = [f"# Statistical tests over {n_runs} runs", "",
             f"α = {alpha}. Bootstrap: {n_boot} resamples, seed {seed}, percentile 95 % intervals. "
             f"Tests compare every ML model with each **tuned** rule variant; all p-values are Holm-corrected "
             f"together ({int(tests['p'].notna().sum()) if not tests.empty else 0} tests). "
             f"A test needs at least {MIN_PAIRS} informative faults; otherwise it is reported as not tested.", "",
             "## Hypotheses", ""]
    if tests.empty:
        lines.append("No ML model and tuned rule variant were evaluated in the same runs.")
    else:
        lines += [f"- {hypothesis_line(t)}" for _, t in tests.iterrows()]
    for dup, first in tests.attrs.get("identical", {}).items():
        lines += ["", f"`{dup}` detected the same faults with the same delays as `{first}` in every run, so its "
                      f"tests are identical to those of `{first}` and are not repeated (nor counted twice in the "
                      f"Holm correction)."]
    lines += ["", "## F1 with 95 % confidence intervals", "",
              "Pooled F1 = F1 of all test windows of all runs together (it differs slightly from the "
              "mean of per-run F1 in `summary_metrics.md`).", "",
              "| Method | Pooled F1 | 95 % CI | Clusters | Interval |", "|---|---:|---:|---:|---|"]
    for _, r in f1_ci.iterrows():
        lines.append(f"| {METHOD_LABELS[r['method']]} | {r['f1_pooled']:.3f} | {_ci(r['f1_ci_low'], r['f1_ci_high'])} | "
                     f"{r['clusters']} | {r['ci_method']} |")
    lines += ["", "## Mean detection delay with 95 % confidence intervals", "",
              "Cluster bootstrap by fault (faults resampled within each run); mean over detected faults.", "",
              "| Method | Detected | Mean delay | 95 % CI |", "|---|---:|---:|---:|"]
    for _, r in d_ci.iterrows():
        mean = "–" if math.isnan(r["mean_delay_s"]) else f"{r['mean_delay_s']:.1f} s"
        lines.append(f"| {METHOD_LABELS[r['method']]} | {r['detected']}/{r['faults']} | {mean} | "
                     f"{_ci(r['delay_ci_low'], r['delay_ci_high'], 1, ' s')} |")
    if not tests.empty:
        lines += ["", "## Test details", "",
                  "Delay differences are rule − ML (positive = ML earlier).", "",
                  "| Method | Rule variant | Test | n | ML only / rule only | Median diff | r | p | p (Holm) | Reject H₀ |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for _, t in tests.iterrows():
            only = f"{t['ml_only']} / {t['rule_only']}" if t["hypothesis"] == "detection" else "–"
            md = f"{t['median_diff_s']:.1f} s" if t["hypothesis"] == "delay" and pd.notna(t["median_diff_s"]) else "–"
            r = f"{t['r']:.2f}" if t.get("r") is not None and not pd.isna(t.get("r")) else "–"
            reject = "not tested" if pd.isna(t["p"]) else ("yes" if t["reject"] else "no")
            lines.append(f"| {METHOD_LABELS[t['method']]} | {t['rule']} | {t['test']} | {t['n']} | {only} | {md} | "
                         f"{r} | {_p(t['p'])} | {_p(t['p_holm'])} | {reject} |")
    return "\n".join(lines) + "\n"
