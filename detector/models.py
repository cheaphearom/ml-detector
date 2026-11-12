"""The detectors compared in the thesis.

All share one interface:

    fit(X, y=None) -> self
    score(X) -> np.ndarray     higher = more anomalous (used for ROC-AUC)
    predict(X) -> np.ndarray   0/1 alert per window

``X`` is the window frame produced by :mod:`detector.features` (one row per window, columns
``window_start``, ``window_end`` and the ``<service>__<feature>`` / ``system__<feature>``
features).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import feature_name, system_name
from .profiles import Profile, get_profile

META_COLUMNS = ("window_start", "window_end", "label", "fault_id")


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every feature column (anything named ``<scope>__<feature>``)."""
    return [c for c in frame.columns if "__" in c and c not in META_COLUMNS]


# ---------------------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleConfig:
    """Thresholds of the traditional monitoring rules (Prometheus-style alerts).

    "gateway" is the profile's entry service (``api-gateway`` for e-commerce, ``pos-api`` for the POS).
    """

    gateway_5xx_ratio: float = 0.05     # > 5 % of gateway responses are 5xx
    gateway_p95_ms: float = 2000.0      # gateway p95 latency above 2 s
    errors_per_service: float = 10.0    # more than 10 ERROR lines in one service per window
    min_requests: int = 10              # 5xx ratio only evaluated with enough traffic
    consecutive: int = 3                # condition must hold for 3 consecutive windows ("for:")
    # extra rules: (event, limit) fires when a window has more than ``limit`` lines of ``event``
    event_limits: tuple[tuple[str, float], ...] = ()

    @classmethod
    def for_profile(cls, profile: Profile | str | None = None) -> "RuleConfig":
        """The default thresholds of a profile (``DETECTOR_PROFILE`` when not given)."""
        p = get_profile(profile)
        return cls(**p.rules, event_limits=p.event_limits)


class RuleBasedDetector:
    """Static threshold rules that must hold for ``consecutive`` windows in a row.

    The continuous score of a window is, over all rules, the maximum of
    ``min(value / threshold)`` across the last ``consecutive`` windows. The rule fires when that
    score exceeds 1, i.e. when every one of those windows is above the threshold.
    """

    name = "rules"
    label = "Rule-based (default)"

    def __init__(self, config: RuleConfig | None = None, step: float = 10.0,
                 services: tuple[str, ...] | None = None, profile: Profile | str | None = None):
        self.profile = get_profile(profile)
        self.config = config or RuleConfig.for_profile(self.profile)
        self.step = step
        self.services = self.profile.services if services is None else tuple(services)
        self.entry_service = self.profile.entry_service

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "RuleBasedDetector":
        return self

    def rule_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per window and rule: observed value divided by its threshold (> 1 = condition true)."""
        c = self.config
        gw = self.entry_service
        ratios = {}
        http_n = _col(X, feature_name(gw, "http_count"))
        r5xx = _col(X, feature_name(gw, "http_5xx_ratio"))
        ratios["gateway_5xx_ratio"] = np.where(http_n >= c.min_requests, r5xx / c.gateway_5xx_ratio, 0.0)
        ratios["gateway_p95_latency"] = _col(X, feature_name(gw, "latency_p95")) / c.gateway_p95_ms
        for svc in self.services:
            ratios[f"errors_{svc}"] = _col(X, feature_name(svc, "error_count")) / c.errors_per_service
        for event, limit in c.event_limits:
            # counts are whole numbers: "more than limit" == count / (limit + 0.5) > 1, also for limit 0
            ratios[f"events_{event}"] = _col(X, system_name(event)) / (limit + 0.5)
        return pd.DataFrame(ratios, index=X.index)

    def sustained_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        ratios = self.rule_ratios(X)
        starts = X["window_start"].to_numpy(dtype=float) if "window_start" in X else None
        return ratios.apply(lambda col: pd.Series(
            sustained_min(col.to_numpy(dtype=float), self.config.consecutive, starts, self.step),
            index=X.index))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.sustained_ratios(X).max(axis=1).to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) > 1.0).astype(int)

    def fired_rules(self, X: pd.DataFrame) -> list[list[str]]:
        sustained = self.sustained_ratios(X)
        return [[name for name, v in row.items() if v > 1.0] for _, row in sustained.iterrows()]


# ---------------------------------------------------------------------------------------
# Rule tuning (training windows only)
# ---------------------------------------------------------------------------------------
RULE_TUNING_GRID: dict[str, tuple] = {
    "consecutive": (1, 2, 3),
    "gateway_5xx_ratio": (0.02, 0.05, 0.10),
    "gateway_p95_ms": (1000.0, 2000.0, 3000.0),
    "errors_per_service": (3.0, 5.0, 10.0),
}


def alert_episodes(pred: np.ndarray, starts: np.ndarray, step: float) -> list[tuple[int, int]]:
    """Maximal runs of consecutive alerting windows as (first index, last index)."""
    episodes = []
    i, n = 0, len(pred)
    while i < n:
        if pred[i]:
            j = i
            while j + 1 < n and pred[j + 1] and abs(starts[j + 1] - starts[j] - step) <= step * 0.5:
                j += 1
            episodes.append((i, j))
            i = j + 1
        else:
            i += 1
    return episodes


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def tune_rules(X: pd.DataFrame, y: np.ndarray, step: float = 10.0, grid: dict[str, tuple] | None = None,
               base: RuleConfig | None = None,
               services: tuple[str, ...] | None = None,
               profile: Profile | str | None = None) -> tuple[RuleConfig, pd.DataFrame]:
    """Grid search of the rule thresholds and ``consecutive`` on **training windows only**.

    Every combination of ``grid`` is scored by window-level F1 against ``y``. The best F1 wins;
    ties go to fewer false-alarm episodes (alert runs without any anomalous window), and any
    remaining tie to the first combination in grid order. Returns the chosen config and the
    full grid table (one row per combination, best first).
    """
    grid = grid or RULE_TUNING_GRID
    base = base or RuleConfig.for_profile(profile)
    y = np.asarray(y, dtype=int)
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    starts = X["window_start"].to_numpy(dtype=float) if "window_start" in X else np.arange(len(X)) * step
    keys = list(grid)
    rows = []
    for order, values in enumerate(itertools.product(*(grid[k] for k in keys))):
        cfg = replace(base, **dict(zip(keys, values)))
        pred = RuleBasedDetector(cfg, step=step, services=services, profile=profile).predict(X) if len(X) else np.zeros(0, int)
        false_eps = sum(1 for a, b in alert_episodes(pred, starts, step) if y[a:b + 1].sum() == 0)
        rows.append({**dict(zip(keys, values)), "f1": _f1(y, pred), "false_alarm_episodes": false_eps,
                     "alerts": int(pred.sum()), "grid_order": order})
    table = pd.DataFrame(rows).sort_values(["f1", "false_alarm_episodes", "grid_order"],
                                           ascending=[False, True, True], kind="stable").reset_index(drop=True)
    best = table.iloc[0]
    chosen = {k: (int(best[k]) if k in ("consecutive", "min_requests") else float(best[k])) for k in keys}
    return replace(base, **chosen), table


# parameters whose smallest grid value is also the smallest meaningful value
NATURAL_LOWER_BOUNDS = {"consecutive": 1}


def grid_edge_check(table: pd.DataFrame, grid: dict[str, tuple]) -> list[dict]:
    """Flags tuned parameters that sit on the edge of the grid **and** matter there.

    For each parameter whose chosen value is the smallest or largest grid value, the best row
    is compared with its neighbour (the next grid value inward, all other parameters equal).
    If the neighbour is worse (lower training F1, or equal F1 with more false-alarm episodes),
    the optimum may lie outside the grid and the grid should be extended in that direction.
    Edges where the neighbour scores the same are not flagged (the parameter is indifferent
    there). ``consecutive = 1`` is a natural lower bound and never flagged.
    """
    best = table.iloc[0]
    keys = list(grid)
    flags = []
    for k in keys:
        values = sorted(grid[k])
        if len(values) < 2:
            continue
        v = best[k]
        if v == values[0] and NATURAL_LOWER_BOUNDS.get(k) != values[0]:
            side, neighbour = "lower", values[1]
        elif v == values[-1]:
            side, neighbour = "upper", values[-2]
        else:
            continue
        same = np.ones(len(table), dtype=bool)
        for other in keys:
            same &= (table[other] == (neighbour if other == k else best[other])).to_numpy()
        row = table[same].iloc[0]
        worse = row["f1"] < best["f1"] or (row["f1"] == best["f1"]
                                           and row["false_alarm_episodes"] > best["false_alarm_episodes"])
        if worse:
            flags.append({"parameter": k, "edge": side, "value": float(v), "neighbour": float(neighbour),
                          "f1": float(best["f1"]), "neighbour_f1": float(row["f1"]),
                          "false_alarm_episodes": int(best["false_alarm_episodes"]),
                          "neighbour_false_alarm_episodes": int(row["false_alarm_episodes"])})
    return flags


class TunedRuleBasedDetector(RuleBasedDetector):
    """The same rules with thresholds and ``consecutive`` chosen by :func:`tune_rules`.

    ``fit`` needs the labels of the training windows; nothing else is learned.
    """

    name = "rules_tuned"
    label = "Rule-based (tuned)"
    consecutive_override: int | None = None

    def __init__(self, step: float = 10.0, grid: dict[str, tuple] | None = None,
                 services: tuple[str, ...] | None = None, profile: Profile | str | None = None):
        super().__init__(None, step=step, services=services, profile=profile)
        self.grid = grid or RULE_TUNING_GRID
        self.tuning_: dict = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "TunedRuleBasedDetector":
        if y is None:
            raise ValueError("rule tuning needs the labels of the training windows")
        tuned, table = tune_rules(X, y, self.step, self.grid, services=self.services, profile=self.profile)
        best = table.iloc[0]
        self.config = tuned if self.consecutive_override is None else replace(
            tuned, consecutive=self.consecutive_override)
        self.tuning_ = {
            "objective": "max window F1 on training windows; ties: fewer false-alarm episodes, then grid order",
            "grid": {k: list(v) for k, v in self.grid.items()},
            "candidates": int(len(table)),
            "tied_best": int(((table["f1"] == best["f1"])
                              & (table["false_alarm_episodes"] == best["false_alarm_episodes"])).sum()),
            "train_windows": int(len(X)),
            "train_f1": float(best["f1"]),
            "train_false_alarm_episodes": int(best["false_alarm_episodes"]),
            "selected": {k: getattr(tuned, k) for k in self.grid},
            "top": table.head(10).drop(columns="grid_order").to_dict(orient="records"),
            "grid_edge": grid_edge_check(table, self.grid),
        }
        return self


class TunedRuleK1Detector(TunedRuleBasedDetector):
    """Tuned thresholds (from the same search) but alerting on a single window (``consecutive=1``)."""

    name = "rules_tuned_k1"
    label = "Rule-based (tuned, k=1)"
    consecutive_override = 1


def sustained_min(values: np.ndarray, k: int, starts: np.ndarray | None = None,
                  step: float = 10.0) -> np.ndarray:
    """Minimum of the last ``k`` consecutive values; 0 until ``k`` consecutive windows exist.

    A gap in ``starts`` (missing windows) resets the run, exactly like a Prometheus ``for:``
    clause that restarts when a sample is missing.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    out = np.zeros(n)
    if n == 0 or k <= 0:
        return out
    run_start = 0
    for i in range(n):
        if starts is not None and i > 0 and abs(starts[i] - starts[i - 1] - step) > step * 0.5:
            run_start = i
        if i - run_start + 1 >= k:
            out[i] = values[i - k + 1:i + 1].min()
    return out


def _col(X: pd.DataFrame, name: str) -> np.ndarray:
    if name in X:
        return X[name].to_numpy(dtype=float)
    return np.zeros(len(X))


# ---------------------------------------------------------------------------------------
# Isolation Forest (unsupervised, trained on normal windows only)
# ---------------------------------------------------------------------------------------
SYMPTOM_SERVICE_FEATURES = (
    "error_ratio", "error_count", "warn_count", "http_5xx", "http_5xx_ratio",
    "latency_mean", "latency_p95", "latency_max", "kafka_p95", "exception_count",
)
SYMPTOM_SYSTEM_FEATURES = get_profile("ecommerce").symptom_system_features   # e-commerce default


def symptom_columns(columns: list[str] | pd.Index, services: tuple[str, ...] | None = None,
                    profile: Profile | str | None = None) -> list[str]:
    """Symptom features: ratios, latencies and error counts - no raw traffic volume.

    System symptoms come from the profile (e.g. saga timeouts, or failed payments for the POS).
    """
    profile = get_profile(profile)
    services = profile.services if services is None else services
    wanted = [feature_name(s, f) for s in services for f in SYMPTOM_SERVICE_FEATURES]
    wanted += [system_name(f) for f in profile.symptom_system_features]
    present = set(columns)
    return [c for c in wanted if c in present]


def _log1p_nonneg(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


class IsolationForestDetector:
    """log1p + StandardScaler + IsolationForest, fitted on normal windows only.

    The alert threshold is the ``percentile``-th percentile of the anomaly scores of the
    training windows, so about (100 - percentile) % of normal windows would alert.
    """

    name = "isolation_forest"
    label = "Isolation Forest"

    def __init__(self, n_estimators: int = 300, percentile: float = 97.0, random_state: int = 42,
                 profile: Profile | str | None = None):
        self.profile = profile
        self.n_estimators = n_estimators
        self.percentile = percentile
        self.random_state = random_state
        self.columns_: list[str] = []
        self.threshold_: float = np.inf
        self.pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "IsolationForestDetector":
        normal = X if y is None else X[np.asarray(y) == 0]
        if len(normal) < 10:
            raise ValueError(f"Isolation Forest needs at least 10 normal windows, got {len(normal)}")
        self.columns_ = symptom_columns(X.columns, profile=self.profile)
        self.pipeline_ = Pipeline([
            ("log1p", FunctionTransformer(_log1p_nonneg)),
            ("scale", StandardScaler()),
            ("forest", IsolationForest(n_estimators=self.n_estimators, contamination="auto",
                                       random_state=self.random_state, n_jobs=-1)),
        ])
        self.pipeline_.fit(normal[self.columns_].to_numpy(dtype=float))
        train_scores = self.score(normal)
        self.threshold_ = float(np.percentile(train_scores, self.percentile))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("IsolationForestDetector is not fitted")
        if len(X) == 0:
            return np.zeros(0)
        return -self.pipeline_.score_samples(X[self.columns_].to_numpy(dtype=float))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)


# ---------------------------------------------------------------------------------------
# Random Forest (supervised)
# ---------------------------------------------------------------------------------------
class RandomForestDetector:
    """Supervised Random Forest on every window feature (300 trees, balanced class weights)."""

    name = "random_forest"
    label = "Random Forest"

    def __init__(self, n_estimators: int = 300, threshold: float = 0.5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.random_state = random_state
        self.columns_: list[str] = []
        self.model_: RandomForestClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "RandomForestDetector":
        if y is None:
            raise ValueError("Random Forest is supervised and needs labels")
        y = np.asarray(y, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("Random Forest needs both normal and anomalous windows in the training data")
        self.columns_ = model_feature_columns(X)
        self.model_ = RandomForestClassifier(n_estimators=self.n_estimators, class_weight="balanced",
                                             min_samples_leaf=2, random_state=self.random_state, n_jobs=-1)
        self.model_.fit(X[self.columns_].to_numpy(dtype=float), y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("RandomForestDetector is not fitted")
        if len(X) == 0:
            return np.zeros(0)
        return self.model_.predict_proba(X[self.columns_].to_numpy(dtype=float))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)

    def feature_importance(self, top: int | None = None) -> list[tuple[str, float]]:
        if self.model_ is None:
            return []
        pairs = sorted(zip(self.columns_, self.model_.feature_importances_), key=lambda p: -p[1])
        pairs = [(name, float(v)) for name, v in pairs]
        return pairs[:top] if top else pairs


# ---------------------------------------------------------------------------------------
# PCA reconstruction error (unsupervised, trained on normal windows only)
# ---------------------------------------------------------------------------------------
class PCADetector:
    """log1p + StandardScaler + PCA on the symptom features, fitted on normal windows only.

    The score is the squared prediction error (SPE): the squared distance between a scaled
    window and its reconstruction from the components that explain ``variance`` of the normal
    training data. The alert threshold follows the same rule as Isolation Forest: the
    ``percentile``-th percentile of the training scores.

    With ``t2`` (on in the POS profile) the score also uses Hotelling's T²: the distance *inside*
    the kept components. SPE alone misses a change along a direction that varies in normal
    traffic (e.g. only the API's latency rising): the reconstruction is then exact. The score is
    ``max(SPE / SPE threshold, T² / T² threshold)``, both thresholds at ``percentile``.
    """

    name = "pca"
    label = "PCA (reconstruction)"

    def __init__(self, variance: float = 0.95, percentile: float = 97.0, profile: Profile | str | None = None,
                 t2: bool | None = None):
        self.profile = profile
        self.t2 = t2
        self.variance = variance
        self.percentile = percentile
        self.columns_: list[str] = []
        self.threshold_: float = np.inf
        self.scaler_: Pipeline | None = None
        self.pca_ = None
        self.n_components_: int = 0
        self.explained_variance_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "PCADetector":
        from sklearn.decomposition import PCA

        normal = X if y is None else X[np.asarray(y) == 0]
        if len(normal) < 10:
            raise ValueError(f"PCA needs at least 10 normal windows, got {len(normal)}")
        self.columns_ = symptom_columns(X.columns, profile=self.profile)
        self.scaler_ = Pipeline([("log1p", FunctionTransformer(_log1p_nonneg)), ("scale", StandardScaler())])
        Z = self.scaler_.fit_transform(normal[self.columns_].to_numpy(dtype=float))
        pca = PCA(n_components=self.variance, svd_solver="full").fit(Z)
        if pca.n_components_ >= Z.shape[1]:
            # every direction kept: the residual would always be 0, so drop the weakest one
            pca = PCA(n_components=max(Z.shape[1] - 1, 1), svd_solver="full").fit(Z)
        self.pca_ = pca
        self.n_components_ = int(pca.n_components_)
        self.explained_variance_ = float(pca.explained_variance_ratio_.sum())
        if self.t2 is None:
            self.t2 = get_profile(self.profile).pca_t2
        self.spe_threshold_ = self.t2_threshold_ = 1.0
        if self.t2:
            self.spe_threshold_ = max(float(np.percentile(self._spe(Z), self.percentile)), 1e-9)
            self.t2_threshold_ = max(float(np.percentile(self._t2(Z), self.percentile)), 1e-9)
        self.threshold_ = float(np.percentile(self._score_z(Z), self.percentile))
        return self

    def _spe(self, Z: np.ndarray) -> np.ndarray:
        recon = self.pca_.inverse_transform(self.pca_.transform(Z))
        return np.sum((Z - recon) ** 2, axis=1)

    def _t2(self, Z: np.ndarray) -> np.ndarray:
        scores = self.pca_.transform(Z)
        return np.sum(scores ** 2 / np.maximum(self.pca_.explained_variance_, 1e-12), axis=1)

    def _score_z(self, Z: np.ndarray) -> np.ndarray:
        if not getattr(self, "t2", False):
            return self._spe(Z)
        return np.maximum(self._spe(Z) / self.spe_threshold_, self._t2(Z) / self.t2_threshold_)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCADetector is not fitted")
        if len(X) == 0:
            return np.zeros(0)
        return self._score_z(self.scaler_.transform(X[self.columns_].to_numpy(dtype=float)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)


# ---------------------------------------------------------------------------------------
# Logistic Regression (supervised)
# ---------------------------------------------------------------------------------------
class LogisticRegressionDetector:
    """log1p + StandardScaler + Logistic Regression on every window feature (balanced weights)."""

    name = "logistic_regression"
    label = "Logistic Regression"

    def __init__(self, C: float = 1.0, threshold: float = 0.5, random_state: int = 42, max_iter: int = 2000):
        self.C = C
        self.threshold = threshold
        self.random_state = random_state
        self.max_iter = max_iter
        self.columns_: list[str] = []
        self.pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "LogisticRegressionDetector":
        from sklearn.linear_model import LogisticRegression

        if y is None:
            raise ValueError("Logistic Regression is supervised and needs labels")
        y = np.asarray(y, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("Logistic Regression needs both normal and anomalous windows in the training data")
        self.columns_ = model_feature_columns(X)
        self.pipeline_ = Pipeline([
            ("log1p", FunctionTransformer(_log1p_nonneg)),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, class_weight="balanced", max_iter=self.max_iter,
                                      random_state=self.random_state)),
        ])
        self.pipeline_.fit(X[self.columns_].to_numpy(dtype=float), y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("LogisticRegressionDetector is not fitted")
        if len(X) == 0:
            return np.zeros(0)
        return self.pipeline_.predict_proba(X[self.columns_].to_numpy(dtype=float))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)

    def coefficients(self, top: int | None = None) -> list[tuple[str, float]]:
        """Standardised coefficients, largest absolute value first."""
        if self.pipeline_ is None:
            return []
        coef = self.pipeline_.named_steps["lr"].coef_[0]
        pairs = sorted(((c, float(v)) for c, v in zip(self.columns_, coef)), key=lambda p: -abs(p[1]))
        return pairs[:top] if top else pairs


RULE_METHODS = (RuleBasedDetector.name, TunedRuleBasedDetector.name, TunedRuleK1Detector.name)
ML_METHODS = (IsolationForestDetector.name, RandomForestDetector.name, PCADetector.name,
              LogisticRegressionDetector.name)
METHOD_ORDER = RULE_METHODS + ML_METHODS
# methods of the live detector (API, notifications); the tuned rules exist only in evaluations
LIVE_METHODS = (RuleBasedDetector.name, IsolationForestDetector.name, RandomForestDetector.name)
METHOD_LABELS = {
    cls.name: cls.label
    for cls in (RuleBasedDetector, TunedRuleBasedDetector, TunedRuleK1Detector,
                IsolationForestDetector, PCADetector, RandomForestDetector, LogisticRegressionDetector)
}
