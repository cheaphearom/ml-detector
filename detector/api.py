"""FastAPI service: training, offline evaluation and live detection.

Endpoints
  POST /train      fit Isolation Forest, PCA (POS profile) and Random Forest (only with labels) on the logs;
                   {"hours": 24} trains on the last 24 hours of logs
  POST /evaluate   run the full offline comparison and write the report files
  GET  /detect     score the most recent window with every available method
  GET  /alerts     alerts raised by the live detection loop
  GET  /scores     anomaly score and threshold of every method for the recent windows
  POST /notify/test  send a test message to the configured Telegram chat
  GET  /health     liveness and model status
  GET  /           monitoring dashboard (see dashboard.py for the endpoints it uses)

A background task tails the log directory every ``LIVE_INTERVAL`` seconds (10 by default),
scores the newest window and records an alert whenever a method fires. Alerts carry an
explanation (suspected service, kind of problem, strongest signals, example log lines) and,
when Telegram is configured, each incident is reported once when it starts and once when it
is resolved.

With DETECTOR_API_KEY set, every endpoint except /health needs the header ``X-API-Key``.

Configuration (environment variables): DETECTOR_PROFILE (ecommerce|pos), DETECTOR_API_KEY, POS_HEALTH_URL,
HEALTH_CHECK_INTERVAL_S, HEALTH_FAIL_AFTER, HEARTBEAT_TIMEOUT_S, SELF_MONITORING, LOG_DIR, LABELS_PATH, RESULTS_DIR, MODEL_DIR, WINDOW,
STEP, GRACE, LIVE_DETECTION, LIVE_INTERVAL, BUFFER_MINUTES, TAIL_FROM_START, SEED,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NOTIFY_METHODS, NOTIFY_RESOLVE_AFTER_S, NOTIFY_TIMEZONE, NOTIFY_MIN_WINDOWS, TELEGRAM_API_URL.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import datetime as dt
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .dashboard import register_dashboard
from .evaluate import evaluate_paths
from .explain import default_baseline, explain, fit_baseline
from .features import (
    add_unseen_templates, build_windows, feature_name, in_fault_mask, known_templates, label_windows, system_name,
)
from .loki import LokiClient, LokiError, LokiTailer, load_logs_from_loki
from .profiles import get_profile
from .models import (
    LIVE_METHODS, RULE_METHODS, IsolationForestDetector, PCADetector, RandomForestDetector, RuleBasedDetector,
)
from .notify import IncidentNotifier, TelegramSender, _clock as notify_clock
from .selfmon import SelfMonitor
from .watch import HealthProbe, HeartbeatWatch, message as check_message
from .parsing import LogTailer, empty_logs, load_experiment_meta, load_labels, load_logs

log = logging.getLogger("detector.api")

MODEL_FILE = "detector.joblib"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # where the logs come from: "files" (LOG_DIR) or "loki" (LOKI_URL, see loki.py)
    log_source: str = field(default_factory=lambda: os.environ.get("LOG_SOURCE", "files").strip().lower() or "files")
    loki_url: str = field(default_factory=lambda: os.environ.get("LOKI_URL", "").strip())
    loki_query: str = field(default_factory=lambda: os.environ.get("LOKI_QUERY", '{job="pos"}').strip() or '{job="pos"}')
    loki_delay_s: float = field(default_factory=lambda: float(os.environ.get("LOKI_DELAY_S") or "10"))
    loki_org_id: str = field(default_factory=lambda: os.environ.get("LOKI_ORG_ID", "").strip())
    loki_username: str = field(default_factory=lambda: os.environ.get("LOKI_USERNAME", "").strip())
    loki_password: str = field(default_factory=lambda: os.environ.get("LOKI_PASSWORD", ""))
    # system checks (watch.py): the API's health endpoint, and the heartbeat lines in its logs
    pos_health_url: str = field(default_factory=lambda: os.environ.get("POS_HEALTH_URL", "").strip())
    health_check_interval_s: float = field(default_factory=lambda: float(os.environ.get("HEALTH_CHECK_INTERVAL_S") or "30"))
    health_fail_after: int = field(default_factory=lambda: int(os.environ.get("HEALTH_FAIL_AFTER") or "2"))
    heartbeat_timeout_s: float = field(default_factory=lambda: float(os.environ.get("HEARTBEAT_TIMEOUT_S") or "120"))
    heartbeat_event: str = field(default_factory=lambda: os.environ.get("HEARTBEAT_EVENT", "heartbeat").strip() or "heartbeat")
    # the detector's own activity to Loki (selfmon.py); default: on when reading from Loki
    self_monitoring: str = field(default_factory=lambda: os.environ.get("SELF_MONITORING", "").strip().lower())
    self_monitoring_job: str = field(default_factory=lambda: os.environ.get("SELF_MONITORING_JOB", "ml-detector").strip() or "ml-detector")
    profile: str = field(default_factory=lambda: os.environ.get("DETECTOR_PROFILE", "ecommerce"))
    log_dir: str = field(default_factory=lambda: os.environ.get("LOG_DIR", "/logs"))
    labels_path: str = field(default_factory=lambda: os.environ.get("LABELS_PATH", "/results/labels.csv"))
    results_dir: str = field(default_factory=lambda: os.environ.get("RESULTS_DIR", "/results/detector"))
    model_dir: str = field(default_factory=lambda: os.environ.get("MODEL_DIR", "/models"))
    window: float = field(default_factory=lambda: float(os.environ.get("WINDOW", "30")))
    step: float = field(default_factory=lambda: float(os.environ.get("STEP", "10")))
    grace: float = field(default_factory=lambda: float(os.environ.get("GRACE", "15")))
    live: bool = field(default_factory=lambda: _env_bool("LIVE_DETECTION", True))
    live_interval: float = field(default_factory=lambda: float(os.environ.get("LIVE_INTERVAL", "10")))
    buffer_minutes: float = field(default_factory=lambda: float(os.environ.get("BUFFER_MINUTES", "10")))
    tail_from_start: bool = field(default_factory=lambda: _env_bool("TAIL_FROM_START", False))
    seed: int = field(default_factory=lambda: int(os.environ.get("SEED", "42")))
    telegram_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", "").strip())
    # empty = the profile's notify_methods
    notify_methods: tuple[str, ...] = field(default_factory=lambda: tuple(
        m.strip() for m in os.environ.get("NOTIFY_METHODS", "").split(",") if m.strip()))
    notify_resolve_after_s: float = field(
        default_factory=lambda: float(os.environ.get("NOTIFY_RESOLVE_AFTER_S") or "60"))
    # 0 = the profile's notify_min_windows
    notify_min_windows: int = field(default_factory=lambda: int(os.environ.get("NOTIFY_MIN_WINDOWS") or "0"))
    notify_timezone: str = field(default_factory=lambda: os.environ.get("NOTIFY_TIMEZONE", "UTC").strip() or "UTC")
    telegram_api_url: str = field(default_factory=lambda: os.environ.get("TELEGRAM_API_URL", "https://api.telegram.org"))
    cors_allow_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()))
    api_key: str = field(default_factory=lambda: os.environ.get("DETECTOR_API_KEY", "").strip())
    min_train_hours: float = field(default_factory=lambda: float(os.environ.get("MIN_TRAIN_HOURS", "24")))


class TrainRequest(BaseModel):
    logs_dir: str | None = Field(None, description="defaults to LOG_DIR")
    labels_path: str | None = Field(None, description="defaults to LABELS_PATH; without labels all windows count as normal")
    start: str | None = None
    end: str | None = None
    hours: float | None = Field(None, gt=0, description="train on the last N hours of logs (ignored with start)")


class EvaluateRequest(BaseModel):
    logs_dir: str | None = None
    labels_path: str | None = None
    out_dir: str | None = None
    window: float | None = None
    step: float | None = None
    train_frac: float = Field(0.6, gt=0.0, lt=1.0)
    grace: float | None = None
    seed: int | None = None
    start: str | None = None
    end: str | None = None
    make_figures: bool = True
    allow_incomplete: bool = False
    tune_rules: bool = Field(False, description="also evaluate the tuned rule variants (rules_tuned, rules_tuned_k1)")
    methods: list[str] | None = Field(None, description="methods to evaluate; default: the live detector's "
                                                        "three methods (plus the tuned rules with tune_rules)")


class DetectorState:
    """Trained models, the rolling log buffer of the live loop, and the alert history."""

    def __init__(self, settings: Settings, send=None):
        """``send`` overrides how notification texts are delivered (used by tests)."""
        self.settings = settings
        self.profile = get_profile(settings.profile)
        self.lock = threading.RLock()
        self.rules = RuleBasedDetector(step=settings.step, profile=self.profile)
        self.live_methods = self.profile.live_methods
        self.iforest: IsolationForestDetector | None = None
        self.pca: PCADetector | None = None
        self.rforest: RandomForestDetector | None = None
        self._model_mtime: float | None = None
        self.known: set[str] = set()
        self.trained_at: str | None = None
        self.training: dict = {}
        self.loki: LokiClient | None = None
        if settings.log_source == "loki":
            self.loki = LokiClient(settings.loki_url, settings.loki_query, settings.loki_org_id,
                                   settings.loki_username, settings.loki_password)
            self.tailer = LokiTailer(self.loki, settings.loki_delay_s,
                                     backfill_s=settings.buffer_minutes * 60.0 if settings.tail_from_start else 0.0)
        elif settings.log_source == "files":
            self.tailer = LogTailer(settings.log_dir, from_end=not settings.tail_from_start)
        else:
            raise ValueError(f"LOG_SOURCE must be files or loki, not {settings.log_source!r}")
        self.buffer = empty_logs()
        self.alerts: collections.deque[dict] = collections.deque(maxlen=1000)
        self.last_alerted: dict[str, float] = {}
        self.last_detection: dict | None = None
        self.history: collections.deque[dict] = collections.deque(maxlen=720)   # 2 h of 10-s windows
        self.polls = 0
        self.baseline: dict = default_baseline(profile=self.profile)
        self.telegram: TelegramSender | None = None
        if settings.telegram_token and settings.telegram_chat_id:
            self.telegram = TelegramSender(settings.telegram_token, settings.telegram_chat_id,
                                           api_url=settings.telegram_api_url)
        if send is None and self.telegram is not None:
            send = self.telegram.send_async
        self.send = send
        self.probe = (HealthProbe(settings.pos_health_url, settings.health_check_interval_s,
                                  fail_after=settings.health_fail_after) if settings.pos_health_url else None)
        self.heartbeat = HeartbeatWatch(settings.heartbeat_timeout_s)
        self.selfmon: SelfMonitor | None = None
        wants_selfmon = settings.self_monitoring in ("1", "true", "yes", "on") or (
            settings.self_monitoring == "" and settings.log_source == "loki")
        if wants_selfmon and settings.loki_url:
            self.selfmon = SelfMonitor(settings.loki_url, settings.self_monitoring_job,
                                       self.loki.headers if self.loki is not None else None)
        self.notify_methods = settings.notify_methods or self.profile.notify_methods
        self.notifier = (IncidentNotifier(send, self.notify_methods, settings.notify_resolve_after_s,
                                          self.profile.system_name, settings.notify_timezone,
                                          settings.notify_min_windows or self.profile.notify_min_windows)
                         if send is not None else None)

    # -- persistence -------------------------------------------------------------------
    def model_path(self) -> Path:
        return Path(self.settings.model_dir) / MODEL_FILE

    def save(self) -> None:
        path = self.model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"iforest": self.iforest, "pca": self.pca, "rforest": self.rforest, "known": self.known,
                     "baseline": self.baseline, "trained_at": self.trained_at, "training": self.training,
                     "window": self.settings.window, "step": self.settings.step, "profile": self.profile.name}, path)

    def load(self) -> bool:
        path = self.model_path()
        if not path.exists():
            return False
        bundle = joblib.load(path)
        if bundle.get("window") != self.settings.window or bundle.get("step") != self.settings.step:
            log.warning("saved model uses a different window/step; ignoring it")
            return False
        if bundle.get("profile", "ecommerce") != self.profile.name:
            log.warning("saved model was trained for profile %s, not %s; ignoring it",
                        bundle.get("profile", "ecommerce"), self.profile.name)
            return False
        with self.lock:
            self.iforest = bundle["iforest"]
            self.pca = bundle.get("pca")
            self.rforest = bundle["rforest"]
            self.known = bundle["known"]
            self.baseline = bundle.get("baseline") or default_baseline(profile=self.profile)
            self.trained_at = bundle["trained_at"]
            self.training = bundle["training"]
            self._model_mtime = path.stat().st_mtime
        return True

    def reload_if_changed(self) -> None:
        """Picks up a model trained by the CLI (``python -m detector.cli train``) in another process."""
        path = self.model_path()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if self._model_mtime is not None and mtime <= self._model_mtime:
            return
        try:
            if self.load():
                log.info("loaded newly trained models from %s", path)
            else:
                self._model_mtime = mtime   # incompatible model: do not try again until it changes
        except Exception:  # noqa: BLE001 - a half-written file is read again next time
            log.exception("could not load %s", path)

    # -- training ----------------------------------------------------------------------
    def train(self, logs_dir: str | None, labels_path: str | None, start: str | None = None,
              end: str | None = None, hours: float | None = None) -> dict:
        """Fits the models on the logs.

        Without labels (the POS) only the unsupervised models are trained, on windows where the
        rules did not fire, so a known incident in the training period is not learned as normal.
        Random Forest needs a labels file with faults.
        """
        s = self.settings
        labels = load_labels(labels_path)
        meta = load_experiment_meta(labels_path)
        start_ts = _ts(start) or _ts(meta.get("load_start"))
        end_ts = _ts(end) or _ts(meta.get("load_end"))
        if logs_dir is None and self.loki is not None:
            # from Loki: the given period, or the last `hours` (default 24) up to LOKI_DELAY_S ago
            end_ts = end_ts or pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=s.loki_delay_s)
            start_ts = start_ts or end_ts - pd.Timedelta(hours=hours or 24.0)
            try:
                logs = load_logs_from_loki(self.loki, start_ts, end_ts)
            except LokiError as e:
                raise ValueError(str(e)) from None
            source = f"Loki ({s.loki_query}, {start_ts:%Y-%m-%d %H:%M} – {end_ts:%H:%M} UTC)"
        else:
            logs_dir = logs_dir or s.log_dir
            logs = load_logs(logs_dir, start=start_ts, end=end_ts)
            source = logs_dir
        if logs.empty:
            raise ValueError(f"no log records found in {source}")
        # heartbeats only show that the logs arrive; like the live loop, leave them out of the features
        logs = logs[logs["event"] != s.heartbeat_event].reset_index(drop=True)
        if logs.empty:
            raise ValueError(f"only heartbeat lines in {source}: no traffic to learn from")
        if hours and start_ts is None:
            logs = logs[logs["t"] >= float(logs["t"].max()) - hours * 3600.0].reset_index(drop=True)
        windows = label_windows(build_windows(logs, s.window, s.step, profile=self.profile), labels, s.grace)
        if len(windows) < 20:
            raise ValueError(f"only {len(windows)} windows - not enough data to train")
        normal_logs = ~in_fault_mask(logs["t"].to_numpy(), labels, s.grace)
        known = known_templates(logs, normal_logs)
        windows = add_unseen_templates(windows, logs, known)
        y = windows["label"].to_numpy()
        has_labels = len(labels) > 0
        # unsupervised models learn "normal": leave out labelled faults, or windows the rules flag
        exclude = y.astype(bool) if has_labels else self.rules.predict(windows).astype(bool)
        y_normal = exclude.astype(int)
        if (y_normal == 0).sum() < 20:
            raise ValueError("fewer than 20 normal windows - not enough data to train")

        iforest = IsolationForestDetector(random_state=s.seed, profile=self.profile).fit(windows, y_normal)
        pca = PCADetector(profile=self.profile).fit(windows, y_normal) if "pca" in self.live_methods else None
        rforest = None
        if has_labels and np.unique(y).size == 2 and "random_forest" in self.live_methods:
            rforest = RandomForestDetector(random_state=s.seed).fit(windows, y)
        span_h = (float(logs["t"].max()) - float(logs["t"].min())) / 3600.0

        baseline = fit_baseline(windows[y_normal == 0], profile=self.profile)
        with self.lock:
            self.iforest, self.pca, self.rforest, self.known = iforest, pca, rforest, known
            self.baseline = baseline or default_baseline(profile=self.profile)
            self.trained_at = dt.datetime.now(dt.timezone.utc).isoformat()
            self.training = {
                "log_records": int(len(logs)), "windows": int(len(windows)),
                "anomalous_windows": int(y.sum()), "faults": int(len(labels)),
                "known_templates": len(known), "isolation_forest_threshold": iforest.threshold_,
                "pca": pca is not None, "random_forest": rforest is not None,
                "excluded_windows": int(exclude.sum()), "hours": round(span_h, 2),
                "first_log": _iso(float(logs["t"].min())), "last_log": _iso(float(logs["t"].max())),
            }
            if span_h < s.min_train_hours - 0.1:
                self.training["warning"] = (f"only {span_h:.1f} hours of logs: train again after at least "
                                            f"{s.min_train_hours:g} hours of normal traffic")
            self.save()
            self._model_mtime = self.model_path().stat().st_mtime
            if self.selfmon is not None:
                self.selfmon.event("model_trained", **{k: v for k, v in self.training.items() if not isinstance(v, (dict, list))})
                self.selfmon.flush()
            return {"trained_at": self.trained_at, **self.training}

    # -- live detection ----------------------------------------------------------------
    def poll(self) -> dict | None:
        """Reads new log lines, scores the newest window and records alerts."""
        s = self.settings
        self.reload_if_changed()
        new = self.tailer.read_new()
        now = time.time()
        if not new.empty:
            # heartbeats prove the logs arrive; they are not part of the anomaly scores
            beats = new["event"] == s.heartbeat_event
            self.heartbeat.saw(new.loc[beats, "t"].to_numpy(), now)
            new = new[~beats].reset_index(drop=True)
        self._run_checks(now)
        with self.lock:
            self.polls += 1
            if not new.empty:
                self.buffer = new if self.buffer.empty else pd.concat([self.buffer, new], ignore_index=True)
                self.buffer = self.buffer.sort_values("t", kind="stable")
                horizon = float(self.buffer["t"].max()) - s.buffer_minutes * 60.0
                self.buffer = self.buffer[self.buffer["t"] >= horizon].reset_index(drop=True)
            result = self._detect_latest()
            if result is not None:
                self.last_detection = result
                if not self.history or self.history[-1]["t"] < result["window_end_epoch"]:
                    self.history.append({
                        "t": result["window_end_epoch"],
                        "methods": {m: {"score": i["score"], "threshold": i["threshold"], "alert": i["alert"]}
                                    for m, i in result["methods"].items()},
                    })
                self._record_alerts(result)
                if self.notifier is not None:
                    change = self.notifier.update(result)
                    if change and self.selfmon is not None:
                        self.selfmon.event("incident_" + change, level="WARN" if change == "opened" else "INFO",
                                           incident=self.notifier.incidents)
            self._self_log(now, len(new), result)
            return result

    # -- system checks (watch.py) -------------------------------------------------------
    def _run_checks(self, now: float) -> None:
        if self.probe is not None and self.probe.due(now):
            change, res = self.probe.run(now)
            if self.selfmon is not None:
                self.selfmon.event("pos_api_check", level="INFO" if res["ok"] else "WARN", url=self.probe.url, **res)
            if change:
                self._check_changed("pos_api", change, res["detail"], self.probe.state.down_since, now)
        read_error = getattr(self.tailer, "last_error", None)
        change, reason = self.heartbeat.check(now, read_error)
        if change:
            self._check_changed("heartbeat", change, reason, self.heartbeat.state.down_since, now)

    def _check_changed(self, kind: str, change: str, detail: str, since: float | None, now: float) -> None:
        entry = self.profile.entry_service
        if change == "down":
            problem = (f"API not responding: {detail}" if kind == "pos_api" else f"no logs: {detail}")
            with self.lock:
                self.alerts.append({
                    "raised_at": dt.datetime.now(dt.timezone.utc).isoformat(), "method": f"{kind}_check",
                    "window_start": _iso(now), "window_end": _iso(now), "score": 1.0, "threshold": 1.0,
                    "explanation": {"suspected_service": entry, "suspected_problem": problem, "top_signals": [],
                                    "notes": [], "example_errors": []},
                })
            log.warning("system check %s: down (%s)", kind, detail)
        else:
            log.info("system check %s: back", kind)
        if self.selfmon is not None:
            self.selfmon.event(f"check_{change}", level="WARN" if change == "down" else "INFO", check=kind, detail=detail)
        if self.send is not None:
            self.send(check_message(kind, change, self.profile.system_name, detail, since,
                                    lambda t: notify_clock(t, self.settings.notify_timezone)))

    def _self_log(self, now: float, lines: int, result: dict | None) -> None:
        if self.selfmon is None:
            return
        fields = {"lines_read": lines, "buffered": int(len(self.buffer)),
                  "read_error": getattr(self.tailer, "last_error", None),
                  "heartbeat_age_s": round(now - self.heartbeat.last_seen, 1) if self.heartbeat.last_seen else None,
                  "incident_open": bool(self.notifier is not None and self.notifier.open_since is not None)}
        if self.probe is not None:
            fields.update(pos_api_ok=not self.probe.state.down and self.probe.state.last_ok is not None,
                          pos_api_ms=self.probe.last_ms)
        if self.telegram is not None:
            fields.update(telegram_pending=self.telegram.pending, telegram_sent=self.telegram.sent,
                          telegram_dropped=self.telegram.dropped)
        if result is not None:
            fields["window_end"] = result["window_end"]
            fields["lag_s"] = round(now - result["window_end_epoch"], 1)
            fields["any_alert"] = any(m["alert"] for m in result["methods"].values())
            for m, info in result["methods"].items():
                fields[f"score_{m}"] = round(float(info["score"]), 4)
                fields[f"threshold_{m}"] = round(float(info["threshold"]), 4)
                fields[f"alert_{m}"] = bool(info["alert"])
        self.selfmon.event("poll", level="WARN" if fields.get("any_alert") else "INFO", **fields)
        self.selfmon.flush()

    def _detect_latest(self) -> dict | None:
        s = self.settings
        # The windows follow the log time. Heartbeats move that clock on when nobody uses the system, so
        # quiet periods become windows without activity (as in training) and incidents can be resolved.
        clock = [float(self.buffer["t"].max())] if not self.buffer.empty else []
        if self.heartbeat.last_heartbeat is not None:
            clock.append(self.heartbeat.last_heartbeat)
        if not clock:
            return None
        t_end = max(clock)
        history = s.window + s.step * (self.rules.config.consecutive + 1)
        recent = self.buffer[self.buffer["t"] >= t_end - history - s.window]
        windows = build_windows(recent, s.window, s.step, start=t_end - history, end=t_end, profile=self.profile)
        if windows.empty:
            return None
        windows = add_unseen_templates(windows, recent, self.known) if self.known else windows
        last = windows.iloc[[-1]]
        methods = {}
        rule_scores = self.rules.score(windows)
        fired = self.rules.fired_rules(windows)[-1]
        methods[RuleBasedDetector.name] = {"score": float(rule_scores[-1]), "threshold": 1.0,
                                           "alert": bool(rule_scores[-1] > 1.0), "fired_rules": fired}
        if self.iforest is not None:
            score = float(self.iforest.score(last)[0])
            methods[IsolationForestDetector.name] = {"score": score, "threshold": self.iforest.threshold_,
                                                     "alert": bool(score > self.iforest.threshold_)}
        if self.pca is not None:
            score = float(self.pca.score(last)[0])
            methods[PCADetector.name] = {"score": score, "threshold": self.pca.threshold_,
                                         "alert": bool(score > self.pca.threshold_)}
        if self.rforest is not None:
            score = float(self.rforest.score(last)[0])
            methods[RandomForestDetector.name] = {"score": score, "threshold": self.rforest.threshold,
                                                  "alert": bool(score >= self.rforest.threshold)}
        row = last.iloc[0]
        explanation = None
        if any(m["alert"] for m in methods.values()):
            explanation = explain(row, self.baseline, recent, profile=self.profile)
        p = self.profile
        key_features = {feature_name(p.entry_service, f): float(row[feature_name(p.entry_service, f)])
                        for f in ("http_count", "http_5xx_ratio", "latency_p95")}
        key_features.update({feature_name(svc, "error_count"): float(row[feature_name(svc, "error_count")])
                             for svc in p.services})
        key_features.update({system_name(f): float(row[system_name(f)])
                             for f in (p.success_ratio.name,) + p.symptom_events + ("unseen_templates",)})
        return {
            "window_start": _iso(row["window_start"]),
            "window_end": _iso(row["window_end"]),
            "window_end_epoch": float(row["window_end"]),
            "methods": methods,
            "features": key_features,
            "explanation": explanation,
        }

    def _record_alerts(self, result: dict) -> None:
        for method, info in result["methods"].items():
            if not info["alert"]:
                continue
            if self.last_alerted.get(method, -1.0) >= result["window_end_epoch"]:
                continue
            self.last_alerted[method] = result["window_end_epoch"]
            alert = {"raised_at": dt.datetime.now(dt.timezone.utc).isoformat(), "method": method,
                     "window_start": result["window_start"], "window_end": result["window_end"],
                     "score": info["score"], "threshold": info["threshold"]}
            if "fired_rules" in info:
                alert["fired_rules"] = info["fired_rules"]
            if result.get("explanation"):
                alert["explanation"] = result["explanation"]
            self.alerts.append(alert)
            if self.selfmon is not None:
                exp = result.get("explanation") or {}
                self.selfmon.event("alert", level="WARN", method=method, score=info["score"], threshold=info["threshold"],
                                   suspected_service=exp.get("suspected_service"), problem=exp.get("suspected_problem"))
            log.warning("anomaly alert: %s score=%.3f window_end=%s", method, info["score"], result["window_end"])

    def status(self) -> dict:
        with self.lock:
            return {
                RuleBasedDetector.name: True,
                IsolationForestDetector.name: self.iforest is not None,
                **({PCADetector.name: self.pca is not None} if PCADetector.name in self.live_methods else {}),
                RandomForestDetector.name: self.rforest is not None,
                "trained_at": self.trained_at,
            }


def _ts(value) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(float(epoch), tz=dt.timezone.utc).isoformat()


def create_app(settings: Settings | None = None, send=None) -> FastAPI:
    """``send`` replaces the Telegram sender (tests pass a function that records messages)."""
    settings = settings or Settings()
    state = DetectorState(settings, send=send)

    async def live_loop() -> None:
        while True:
            try:
                await asyncio.to_thread(state.poll)
            except Exception:  # noqa: BLE001 - the loop must survive malformed input
                log.exception("live detection poll failed")
            await asyncio.sleep(settings.live_interval)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if state.load():
                log.info("loaded trained models from %s", state.model_path())
        except Exception:  # noqa: BLE001
            log.exception("could not load saved models")
        task = asyncio.create_task(live_loop()) if settings.live else None
        yield
        if state.telegram is not None:
            state.telegram.close()
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Log anomaly detector", version=__version__, lifespan=lifespan)
    # CORS for the Flutter web app (mobile/), which runs on another origin; native apps and the
    # dashboard (same origin) do not need it. Restrict with CORS_ALLOW_ORIGINS="http://host:port,...".
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_allow_origins,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
    app.state.detector = state

    if settings.api_key:
        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            import hmac
            if request.url.path != "/health" and not hmac.compare_digest(
                    request.headers.get("x-api-key", ""), settings.api_key):
                return JSONResponse({"detail": "missing or wrong X-API-Key"}, status_code=401)
            return await call_next(request)

    register_dashboard(app, state, settings)

    @app.get("/health")
    def health() -> dict:
        last = state.last_detection
        source = {"type": settings.log_source}
        if state.loki is not None:
            source.update(url=state.loki.url, query=state.loki.query, delay_s=settings.loki_delay_s,
                          error=state.tailer.last_error)
        return {"status": "UP", "version": __version__, "profile": state.profile.name, "live_detection": settings.live,
                "log_source": source,
                "models": state.status(), "buffered_records": int(len(state.buffer)), "polls": state.polls,
                "last_window_end": last["window_end"] if last else None,
                "telegram_notifications": state.telegram is not None, "notify_methods": list(state.notify_methods),
                "checks": {"pos_api": ({**state.probe.state.view(), "url": state.probe.url, "ms": state.probe.last_ms}
                                       if state.probe is not None else None),
                           "heartbeat": state.heartbeat.view(time.time())},
                "self_monitoring": state.selfmon.view() if state.selfmon is not None else None,
                "telegram": ({"pending": state.telegram.pending, "sent": state.telegram.sent,
                              "dropped": state.telegram.dropped, "last_error": state.telegram.last_error}
                             if state.telegram is not None else None),
                "training": {k: v for k, v in state.training.items() if k in ("hours", "windows", "warning")}}

    @app.post("/notify/test")
    def notify_test() -> dict:
        if state.telegram is None:
            raise HTTPException(status_code=400,
                                detail="Telegram is not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        try:
            where = f" for the {state.profile.system_name}" if state.profile.system_name else ""
            state.telegram.send(f"✅ Test message from the log anomaly detector{where}. Telegram alerts are working.")
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        return {"sent": True}

    @app.post("/train")
    def train(request: TrainRequest | None = None) -> dict:
        request = request or TrainRequest()
        try:
            return state.train(request.logs_dir or (None if state.loki else settings.log_dir),
                               request.labels_path or settings.labels_path,
                               request.start, request.end, request.hours)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/evaluate")
    def evaluate(request: EvaluateRequest | None = None) -> dict:
        request = request or EvaluateRequest()
        labels_path = request.labels_path or settings.labels_path
        if not Path(labels_path).exists():
            raise HTTPException(status_code=400, detail=f"labels file not found: {labels_path}")
        out_dir = request.out_dir or settings.results_dir
        try:
            results = evaluate_paths(
                request.logs_dir or settings.log_dir, labels_path, out_dir=out_dir,
                window=request.window or settings.window, step=request.step or settings.step,
                train_frac=request.train_frac,
                grace=request.grace if request.grace is not None else settings.grace,
                seed=request.seed if request.seed is not None else settings.seed,
                start=request.start, end=request.end, make_figures=request.make_figures,
                tune_rules=request.tune_rules or bool(request.methods),
                methods=request.methods or (list(LIVE_METHODS) + (list(RULE_METHODS) if request.tune_rules else [])),
                allow_incomplete=request.allow_incomplete)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "out_dir": out_dir,
            "outputs": results.get("outputs", []),
            "dataset": {k: v for k, v in results["dataset"].items() if k != "experiment"},
            "metrics": results["metrics"],
            "detection_delay": results["detection_delay"],
            "comparison_vs_rules": results["comparison_vs_rules"],
            "by_fault_type": results["by_fault_type"],
        }

    @app.get("/detect")
    def detect() -> dict:
        result = state.poll() if not settings.live else (state.last_detection or state.poll())
        if result is None:
            return {"status": "insufficient_data",
                    "detail": f"need at least {settings.window:.0f} s of logs in {settings.log_dir}"}
        anomaly = any(m["alert"] for m in result["methods"].values())
        return {"status": "anomaly" if anomaly else "normal",
                **{k: v for k, v in result.items() if k != "window_end_epoch"}}

    @app.get("/scores")
    def scores(minutes: float = 30.0) -> dict:
        """Score, threshold and alert of every method per window (one point per live poll)."""
        minutes = max(1.0, min(minutes, 120.0))
        with state.lock:
            points = list(state.history)
        if points:
            cutoff = points[-1]["t"] - minutes * 60.0
            points = [p for p in points if p["t"] >= cutoff]
        available = [m for m, ready in state.status().items() if m != "trained_at" and ready]
        return {"methods": available,
                "points": [{"time": _iso(p["t"]), **p} for p in points]}

    @app.get("/alerts")
    def alerts(limit: int = 50) -> dict:
        limit = max(1, min(limit, 1000))
        with state.lock:
            items = list(state.alerts)[-limit:]
        return {"count": len(items), "alerts": list(reversed(items))}

    return app


app = create_app()
