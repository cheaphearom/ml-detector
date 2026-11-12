"""Human-readable explanation of an anomalous window.

The detectors only answer *whether* a window is abnormal. This module answers *where* and
*what kind*: it compares each symptom feature with its normal level, names the service whose
symptoms deviate most, guesses the kind of problem from the mix of latency and error
signals, and attaches real error lines from the logs of that window.

Normal levels (a baseline) are learned from normal training windows; before any training a
conservative built-in default is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_name, system_name
from .profiles import Profile, get_profile

# Symptom features per service that the explanation looks at.
LATENCY_FEATURES = ("latency_p95", "latency_mean", "kafka_p95")
ERROR_FEATURES = ("error_count", "exception_count", "http_5xx", "error_ratio", "http_5xx_ratio")
SERVICE_SYMPTOMS = LATENCY_FEATURES + ERROR_FEATURES + ("warn_count",)
# e-commerce defaults; the functions below take them from the profile
SYSTEM_SYMPTOMS = get_profile("ecommerce").symptom_system_features
LOWER_IS_WORSE = {system_name("saga_success_ratio")}

SIGNIFICANT_Z = 3.0
BACKENDS = get_profile("ecommerce").backends

READABLE = {
    "latency_p95": "p95 latency (ms)",
    "latency_mean": "mean latency (ms)",
    "latency_max": "max latency (ms)",
    "kafka_p95": "Kafka message p95 (ms)",
    "error_count": "ERROR log lines",
    "exception_count": "exceptions",
    "http_5xx": "HTTP 5xx responses",
    "error_ratio": "share of ERROR lines",
    "http_5xx_ratio": "share of 5xx responses",
    "warn_count": "WARN log lines",
    "saga_timeout": "saga timeouts",
    "saga_late_reply": "late saga replies",
    "saga_success_ratio": "saga success ratio",
    "unseen_templates": "new, never-seen log messages",
}


def readable(name: str, profile: Profile | str | None = None) -> str:
    """Human name of a feature (without the ``<scope>__`` prefix)."""
    return get_profile(profile).readable.get(name) or READABLE.get(name, name)


def lower_is_worse(profile: Profile | str | None = None) -> set[str]:
    return {system_name(get_profile(profile).success_ratio.name)}


def explanation_columns(services: tuple[str, ...] | None = None, profile: Profile | str | None = None) -> list[str]:
    profile = get_profile(profile)
    services = profile.services if services is None else services
    cols = [feature_name(s, f) for s in services for f in SERVICE_SYMPTOMS]
    return cols + [system_name(f) for f in profile.symptom_system_features]


def _floor(column: str) -> float:
    """Smallest spread treated as meaningful, so a constant-zero feature does not explode."""
    name = column.split("__", 1)[1]
    if name in ("latency_p95", "latency_mean", "latency_max", "kafka_p95"):
        return 25.0
    if name.endswith("ratio"):
        return 0.03
    return 1.0


def default_baseline(services: tuple[str, ...] | None = None,
                     profile: Profile | str | None = None) -> dict[str, tuple[float, float]]:
    """A conservative normal level used before the detector has been trained."""
    profile = get_profile(profile)
    base = {}
    for col in explanation_columns(services, profile):
        name = col.split("__", 1)[1]
        if name in ("latency_p95", "latency_mean", "kafka_p95"):
            base[col] = (50.0, 100.0)
        elif name == profile.success_ratio.name:
            base[col] = (0.9, 0.1)
        elif name.endswith("ratio"):
            base[col] = (0.0, 0.03)
        else:
            base[col] = (0.0, 2.0)
    return base


def fit_baseline(normal_windows: pd.DataFrame,
                 profile: Profile | str | None = None) -> dict[str, tuple[float, float]]:
    """Median and spread of every symptom feature over normal windows."""
    base = {}
    for col in explanation_columns(profile=profile):
        if col not in normal_windows or normal_windows.empty:
            continue
        values = normal_windows[col].to_numpy(dtype=float)
        median = float(np.median(values))
        spread = float(np.std(values))
        base[col] = (median, max(spread, _floor(col)))
    return base


def deviation(column: str, value: float, baseline: dict[str, tuple[float, float]],
              profile: Profile | str | None = None) -> float:
    """How many 'normal spreads' a value is in the bad direction (never negative)."""
    median, scale = baseline.get(column, (0.0, _floor(column)))
    z = (value - median) / scale
    if column in LOWER_IS_WORSE or column in lower_is_worse(profile):
        z = -z
    return max(0.0, float(z))


def _fmt(column: str, value: float) -> str:
    name = column.split("__", 1)[1]
    if name.endswith("ratio"):
        return f"{value:.2f}"
    if name in ("latency_p95", "latency_mean", "latency_max", "kafka_p95"):
        return f"{value:.0f} ms"
    return f"{value:.0f}"


def explain(window: pd.Series, baseline: dict[str, tuple[float, float]] | None = None,
            logs: pd.DataFrame | None = None, max_signals: int = 5, max_errors: int = 3,
            profile: Profile | str | None = None) -> dict:
    """Explains one window (a row of the feature frame)."""
    profile = get_profile(profile)
    entry = profile.entry_service
    baseline = baseline or default_baseline(profile=profile)
    z = {col: deviation(col, float(window[col]), baseline, profile)
         for col in explanation_columns(profile=profile) if col in window.index}

    def service_scores(service: str) -> tuple[float, float, float]:
        lat = max(z.get(feature_name(service, f), 0.0) for f in LATENCY_FEATURES)
        err = max(z.get(feature_name(service, f), 0.0) for f in ERROR_FEATURES)
        total = sum(z.get(feature_name(service, f), 0.0) for f in SERVICE_SYMPTOMS)
        return lat, err, total

    scores = {s: service_scores(s) for s in profile.services}
    best_backend = max(profile.backends, key=lambda s: scores[s][2])
    if scores[best_backend][2] >= SIGNIFICANT_Z:
        suspect = best_backend
    elif scores[entry][2] >= SIGNIFICANT_Z:
        suspect = entry
    else:
        suspect = None

    if suspect is not None:
        lat, err, _ = scores[suspect]
        # a gateway in front of the backends sees their slowness, so use it to confirm a latency problem
        if suspect != entry and entry not in profile.backends:
            lat = max(lat, scores[entry][0])
        if lat >= SIGNIFICANT_Z and err >= SIGNIFICANT_Z:
            kind = "slow responses and errors (database or connection problem, like DB_TIMEOUT)"
        elif lat >= SIGNIFICANT_Z:
            kind = "slow responses (like LATENCY)"
        elif err >= SIGNIFICANT_Z:
            kind = "errors / exceptions (like ERROR)"
        else:
            kind = "unusual behaviour without one clear signal"
    else:
        kind = "unusual combination of small changes (no single service stands out)"

    signals = sorted(((col, v) for col, v in z.items() if v >= 1.0), key=lambda p: -p[1])[:max_signals]
    top = []
    for col, v in signals:
        scope, name = col.split("__", 1)
        median = baseline.get(col, (0.0, 0.0))[0]
        top.append({
            "feature": col,
            "text": f"{scope}: {readable(name, profile)} = {_fmt(col, float(window[col]))} "
                    f"(normal ≈ {_fmt(col, median)})",
            "deviation": round(v, 1),
        })

    notes = []
    if float(window.get(system_name("unseen_templates"), 0.0)) > 0:
        notes.append("new log messages appeared that never occurred during normal operation")
    for event, note in profile.notes.items():
        if float(window.get(system_name(event), 0.0)) > 0:
            notes.append(note)

    return {
        "suspected_service": suspect,
        "suspected_problem": kind,
        "top_signals": top,
        "notes": notes,
        "example_errors": example_errors(logs, float(window["window_start"]), float(window["window_end"]),
                                         suspect, max_errors) if logs is not None else [],
    }


def example_errors(logs: pd.DataFrame, start: float, end: float, service: str | None,
                   limit: int = 3) -> list[str]:
    """Distinct ERROR (or else WARN) messages from the window, preferring the suspected service."""
    if logs is None or logs.empty:
        return []
    in_window = logs[(logs["t"] >= start) & (logs["t"] < end)]
    for level in ("ERROR", "WARN"):
        rows = in_window[in_window["level"] == level]
        rows = rows[rows["event"] != "http_request"] if (rows["event"] != "http_request").any() else rows
        if service is not None and (rows["service"] == service).any():
            rows = rows[rows["service"] == service]
        if rows.empty:
            continue
        rows = rows.drop_duplicates(subset="template", keep="last").tail(limit)
        return [f"[{r.service}] {_short(r.message)}" for r in rows.itertuples(index=False)]
    return []


def _short(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"
