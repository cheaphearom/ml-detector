"""Sliding-window features computed from parsed logs.

Windows are ``window`` seconds long and start every ``step`` seconds (30 s / 10 s by default).
Per service the features are:

    log_count, error_count, warn_count, error_ratio,
    http_count, http_4xx, http_5xx, http_5xx_ratio,
    latency_mean, latency_p95, latency_max      (event=http_request durationMs)
    kafka_p95                                   (event=message_processed|message_failed durationMs)
    exception_count                             (records with a stack trace)
    distinct_templates

System-wide features (prefix ``system__``) come from the detector profile (see profiles.py):
a count per business event, a success ratio and ``unseen_templates``. For the e-commerce profile:

    saga_started, saga_completed, saga_failed, saga_compensating, saga_compensated,
    saga_timeout, saga_late_reply, saga_success_ratio, unseen_templates

and for the POS profile: sale_completed, sale_failed, payment_failed, …, sale_success_ratio,
unseen_templates. Services also come from the profile.

``unseen_templates`` counts WARN/ERROR lines whose template never occurred in the normal
training data; it is filled in by :func:`add_unseen_templates` once the training period is known.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .profiles import Profile, get_profile

# The e-commerce defaults, kept for code that imports them (evaluation, synthetic logs).
# Everything else takes the services and system features from the active profile.
_ECOMMERCE = get_profile("ecommerce")
SERVICES: tuple[str, ...] = _ECOMMERCE.services

SERVICE_FEATURES: tuple[str, ...] = (
    "log_count", "error_count", "warn_count", "error_ratio",
    "http_count", "http_4xx", "http_5xx", "http_5xx_ratio",
    "latency_mean", "latency_p95", "latency_max",
    "kafka_p95", "exception_count", "distinct_templates",
)

SAGA_EVENTS: tuple[str, ...] = _ECOMMERCE.business_events
SYSTEM_FEATURES: tuple[str, ...] = _ECOMMERCE.system_features

KAFKA_EVENTS = ("message_processed", "message_failed")


def feature_name(service: str, feature: str) -> str:
    return f"{service}__{feature}"


def system_name(feature: str) -> str:
    return f"system__{feature}"


def feature_columns(services: Sequence[str] | None = None, profile: Profile | str | None = None) -> list[str]:
    profile = get_profile(profile)
    services = profile.services if services is None else services
    cols = [feature_name(s, f) for s in services for f in SERVICE_FEATURES]
    cols += [system_name(f) for f in profile.system_features]
    return cols


def window_starts(t_min: float, t_max: float, window: float, step: float) -> np.ndarray:
    """Start times of every complete window inside [t_min, t_max], aligned to multiples of step."""
    first = np.floor(t_min / step) * step
    last = t_max - window
    if last < first:
        return np.empty(0)
    return np.arange(first, last + 1e-9, step)


class _Counter:
    """Fast counts of events in [a, b) via sorted times and searchsorted."""

    def __init__(self, times: np.ndarray):
        self.times = np.sort(np.asarray(times, dtype=float))

    def count(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        return (np.searchsorted(self.times, ends, side="left")
                - np.searchsorted(self.times, starts, side="left")).astype(float)


def _slices(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.searchsorted(times, starts, side="left"), np.searchsorted(times, ends, side="left")


def build_windows(logs: pd.DataFrame, window: float = 30.0, step: float = 10.0,
                  start: float | None = None, end: float | None = None,
                  services: Sequence[str] | None = None, profile: Profile | str | None = None) -> pd.DataFrame:
    """Computes one feature row per sliding window.

    ``start``/``end`` (epoch seconds) bound the windows; by default the span of the logs.
    ``profile`` (default: ``DETECTOR_PROFILE``) gives the services and the system features.
    The ``system__unseen_templates`` column is 0 until :func:`add_unseen_templates` is called.
    """
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    profile = get_profile(profile)
    services = profile.services if services is None else tuple(services)
    cols = ["window_start", "window_end"] + feature_columns(services, profile)
    if logs.empty and (start is None or end is None):
        return pd.DataFrame(columns=cols, dtype=float)
    t_min = float(logs["t"].min()) if start is None else float(start)
    t_max = float(logs["t"].max()) if end is None else float(end)
    starts = window_starts(t_min, t_max, window, step)
    if starts.size == 0:
        return pd.DataFrame(columns=cols, dtype=float)
    ends = starts + window
    out: dict[str, np.ndarray] = {"window_start": starts, "window_end": ends}

    logs = logs.sort_values("t", kind="stable")
    for svc in services:
        s = logs[logs["service"] == svc]
        t = s["t"].to_numpy(dtype=float)
        level = s["level"].to_numpy(dtype=object)
        n = _Counter(t).count(starts, ends)
        errors = _Counter(t[level == "ERROR"]).count(starts, ends)
        warns = _Counter(t[level == "WARN"]).count(starts, ends)
        stacks = _Counter(t[s["has_stack"].to_numpy(dtype=bool)]).count(starts, ends)

        http = s[s["event"] == "http_request"]
        http_t = http["t"].to_numpy(dtype=float)
        http_status = http["status"].to_numpy(dtype=float)
        http_dur = http["durationMs"].to_numpy(dtype=float)
        http_n = _Counter(http_t).count(starts, ends)
        http_4xx = _Counter(http_t[(http_status >= 400) & (http_status < 500)]).count(starts, ends)
        http_5xx = _Counter(http_t[http_status >= 500]).count(starts, ends)

        kafka = s[s["event"].isin(KAFKA_EVENTS)]
        kafka_t = kafka["t"].to_numpy(dtype=float)
        kafka_dur = kafka["durationMs"].to_numpy(dtype=float)

        codes, _ = pd.factorize(s["template"])
        lat_mean = np.zeros(starts.size)
        lat_p95 = np.zeros(starts.size)
        lat_max = np.zeros(starts.size)
        k_p95 = np.zeros(starts.size)
        distinct = np.zeros(starts.size)

        h_lo, h_hi = _slices(http_t, starts, ends)
        k_lo, k_hi = _slices(kafka_t, starts, ends)
        a_lo, a_hi = _slices(t, starts, ends)
        for i in range(starts.size):
            if h_hi[i] > h_lo[i]:
                d = http_dur[h_lo[i]:h_hi[i]]
                d = d[~np.isnan(d)]
                if d.size:
                    lat_mean[i] = d.mean()
                    lat_p95[i] = np.percentile(d, 95)
                    lat_max[i] = d.max()
            if k_hi[i] > k_lo[i]:
                d = kafka_dur[k_lo[i]:k_hi[i]]
                d = d[~np.isnan(d)]
                if d.size:
                    k_p95[i] = np.percentile(d, 95)
            if a_hi[i] > a_lo[i]:
                distinct[i] = np.unique(codes[a_lo[i]:a_hi[i]]).size

        with np.errstate(divide="ignore", invalid="ignore"):
            error_ratio = np.where(n > 0, errors / np.maximum(n, 1), 0.0)
            ratio_5xx = np.where(http_n > 0, http_5xx / np.maximum(http_n, 1), 0.0)

        values = {
            "log_count": n, "error_count": errors, "warn_count": warns, "error_ratio": error_ratio,
            "http_count": http_n, "http_4xx": http_4xx, "http_5xx": http_5xx, "http_5xx_ratio": ratio_5xx,
            "latency_mean": lat_mean, "latency_p95": lat_p95, "latency_max": lat_max,
            "kafka_p95": k_p95, "exception_count": stacks, "distinct_templates": distinct,
        }
        for f in SERVICE_FEATURES:
            out[feature_name(svc, f)] = values[f]

    events = logs["event"].to_numpy(dtype=object)
    t_all = logs["t"].to_numpy(dtype=float)
    for ev in profile.business_events:
        out[system_name(ev)] = _Counter(t_all[events == ev]).count(starts, ends)
    ratio = profile.success_ratio
    num = sum((_event_counts(out, ev, t_all, events, starts, ends) for ev in ratio.numerator), np.zeros(starts.size))
    den = sum((_event_counts(out, ev, t_all, events, starts, ends) for ev in ratio.denominator), np.zeros(starts.size))
    out[system_name(ratio.name)] = np.clip(np.where(den > 0, num / np.maximum(den, 1), 1.0), 0.0, 2.0)
    out[system_name("unseen_templates")] = np.zeros(starts.size)
    return pd.DataFrame(out, columns=cols)


def _event_counts(out: dict, event: str, t_all: np.ndarray, events: np.ndarray,
                  starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Per-window count of ``event`` (reuses the business-event column when there is one)."""
    col = system_name(event)
    if col in out:
        return out[col]
    return _Counter(t_all[events == event]).count(starts, ends)


def known_templates(logs: pd.DataFrame, normal_mask: np.ndarray | pd.Series | None = None) -> set[str]:
    """Templates that occur in the (normal) training logs."""
    if normal_mask is not None:
        logs = logs[np.asarray(normal_mask, dtype=bool)]
    return set(logs["template"].unique())


UNSEEN_LEVELS = ("WARN", "ERROR")


def add_unseen_templates(windows: pd.DataFrame, logs: pd.DataFrame, known: Iterable[str]) -> pd.DataFrame:
    """Fills ``system__unseen_templates``: WARN/ERROR lines whose template is not in ``known``.

    Only warnings and errors count. A new INFO message (for example a request type the training
    traffic never made) is a change in usage, not a failure, and counting it caused false alarms
    as soon as anyone used the system differently from the load generator.
    """
    windows = windows.copy()
    col = system_name("unseen_templates")
    if windows.empty:
        return windows
    known = set(known)
    mask = ~logs["template"].isin(known) & logs["level"].isin(UNSEEN_LEVELS)
    unseen_t = logs.loc[mask, "t"].to_numpy(dtype=float)
    windows[col] = _Counter(unseen_t).count(windows["window_start"].to_numpy(),
                                            windows["window_end"].to_numpy())
    return windows


def fault_intervals(labels: pd.DataFrame, grace: float = 15.0) -> list[tuple[float, float, str]]:
    """(start, end + grace, fault_id) in epoch seconds for every labelled fault."""
    result = []
    for row in labels.itertuples(index=False):
        s = row.start.timestamp()
        e = row.end.timestamp() + grace
        result.append((s, e, row.fault_id))
    return result


def label_windows(windows: pd.DataFrame, labels: pd.DataFrame, grace: float = 15.0) -> pd.DataFrame:
    """Adds ``label`` (1 = window overlaps [fault start, fault end + grace]) and ``fault_id``."""
    windows = windows.copy()
    y = np.zeros(len(windows), dtype=int)
    fault_ids = np.array([""] * len(windows), dtype=object)
    if len(windows):
        ws = windows["window_start"].to_numpy()
        we = windows["window_end"].to_numpy()
        for s, e, fid in fault_intervals(labels, grace):
            overlap = (ws < e) & (we > s)
            fault_ids[overlap & (y == 0)] = fid
            y[overlap] = 1
    windows["label"] = y
    windows["fault_id"] = fault_ids
    return windows


def in_fault_mask(t: np.ndarray, labels: pd.DataFrame, grace: float = 15.0) -> np.ndarray:
    """True for timestamps inside any [fault start, fault end + grace]."""
    mask = np.zeros(len(t), dtype=bool)
    for s, e, _ in fault_intervals(labels, grace):
        mask |= (t >= s) & (t <= e)
    return mask
