"""FastAPI service: training, offline evaluation and live detection.

Endpoints
  POST /train      fit Isolation Forest (and Random Forest when labels exist) on the logs
  POST /evaluate   run the full offline comparison and write the report files
  GET  /detect     score the most recent window with every available method
  GET  /alerts     alerts raised by the live detection loop
  POST /notify/test  send a test message to the configured Telegram chat
  GET  /health     liveness and model status
  GET  /           monitoring dashboard (see dashboard.py for the endpoints it uses)

A background task tails the log directory every ``LIVE_INTERVAL`` seconds (10 by default),
scores the newest window and records an alert whenever a method fires. Alerts carry an
explanation (suspected service, kind of problem, strongest signals, example log lines) and,
when Telegram is configured, each incident is reported once when it starts and once when it
is resolved.

Configuration (environment variables): DETECTOR_PROFILE (ecommerce|pos), LOG_DIR, LABELS_PATH, RESULTS_DIR, MODEL_DIR, WINDOW,
STEP, GRACE, LIVE_DETECTION, LIVE_INTERVAL, BUFFER_MINUTES, TAIL_FROM_START, SEED,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NOTIFY_METHODS, NOTIFY_RESOLVE_AFTER_S.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import datetime as dt
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .dashboard import register_dashboard
from .evaluate import evaluate_paths
from .explain import default_baseline, explain, fit_baseline
from .features import (
    add_unseen_templates, build_windows, feature_name, in_fault_mask, known_templates, label_windows, system_name,
)
from .profiles import get_profile
from .models import LIVE_METHODS, RULE_METHODS, IsolationForestDetector, RandomForestDetector, RuleBasedDetector
from .notify import IncidentNotifier, TelegramSender
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
    notify_methods: tuple[str, ...] = field(default_factory=lambda: tuple(
        m.strip() for m in os.environ.get("NOTIFY_METHODS", ",".join(LIVE_METHODS)).split(",") if m.strip()))
    notify_resolve_after_s: float = field(
        default_factory=lambda: float(os.environ.get("NOTIFY_RESOLVE_AFTER_S", "60")))
    cors_allow_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()))


class TrainRequest(BaseModel):
    logs_dir: str | None = Field(None, description="defaults to LOG_DIR")
    labels_path: str | None = Field(None, description="defaults to LABELS_PATH; without labels all windows count as normal")
    start: str | None = None
    end: str | None = None


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
        self.iforest: IsolationForestDetector | None = None
        self.rforest: RandomForestDetector | None = None
        self.known: set[str] = set()
        self.trained_at: str | None = None
        self.training: dict = {}
        self.tailer = LogTailer(settings.log_dir, from_end=not settings.tail_from_start)
        self.buffer = empty_logs()
        self.alerts: collections.deque[dict] = collections.deque(maxlen=1000)
        self.last_alerted: dict[str, float] = {}
        self.last_detection: dict | None = None
        self.history: collections.deque[dict] = collections.deque(maxlen=720)   # 2 h of 10-s windows
        self.polls = 0
        self.baseline: dict = default_baseline(profile=self.profile)
        self.telegram: TelegramSender | None = None
        if settings.telegram_token and settings.telegram_chat_id:
            self.telegram = TelegramSender(settings.telegram_token, settings.telegram_chat_id)
        if send is None and self.telegram is not None:
            send = self.telegram.send_async
        self.notifier = (IncidentNotifier(send, settings.notify_methods, settings.notify_resolve_after_s)
                         if send is not None else None)

    # -- persistence -------------------------------------------------------------------
    def model_path(self) -> Path:
        return Path(self.settings.model_dir) / MODEL_FILE

    def save(self) -> None:
        path = self.model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"iforest": self.iforest, "rforest": self.rforest, "known": self.known,
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
            self.rforest = bundle["rforest"]
            self.known = bundle["known"]
            self.baseline = bundle.get("baseline") or default_baseline(profile=self.profile)
            self.trained_at = bundle["trained_at"]
            self.training = bundle["training"]
        return True

    # -- training ----------------------------------------------------------------------
    def train(self, logs_dir: str, labels_path: str | None, start: str | None = None,
              end: str | None = None) -> dict:
        s = self.settings
        labels = load_labels(labels_path)
        meta = load_experiment_meta(labels_path)
        start_ts = _ts(start) or _ts(meta.get("load_start"))
        end_ts = _ts(end) or _ts(meta.get("load_end"))
        logs = load_logs(logs_dir, start=start_ts, end=end_ts)
        if logs.empty:
            raise ValueError(f"no log records found in {logs_dir}")
        windows = label_windows(build_windows(logs, s.window, s.step, profile=self.profile), labels, s.grace)
        if len(windows) < 20:
            raise ValueError(f"only {len(windows)} windows - not enough data to train")
        normal_logs = ~in_fault_mask(logs["t"].to_numpy(), labels, s.grace)
        known = known_templates(logs, normal_logs)
        windows = add_unseen_templates(windows, logs, known)
        y = windows["label"].to_numpy()

        iforest = IsolationForestDetector(random_state=s.seed, profile=self.profile).fit(windows, y)
        rforest = None
        if np.unique(y).size == 2:
            rforest = RandomForestDetector(random_state=s.seed).fit(windows, y)

        baseline = fit_baseline(windows[y == 0], profile=self.profile)
        with self.lock:
            self.iforest, self.rforest, self.known = iforest, rforest, known
            self.baseline = baseline or default_baseline(profile=self.profile)
            self.trained_at = dt.datetime.now(dt.timezone.utc).isoformat()
            self.training = {
                "log_records": int(len(logs)), "windows": int(len(windows)),
                "anomalous_windows": int(y.sum()), "faults": int(len(labels)),
                "known_templates": len(known), "isolation_forest_threshold": iforest.threshold_,
                "random_forest": rforest is not None,
            }
            self.save()
            return {"trained_at": self.trained_at, **self.training}

    # -- live detection ----------------------------------------------------------------
    def poll(self) -> dict | None:
        """Reads new log lines, scores the newest window and records alerts."""
        s = self.settings
        new = self.tailer.read_new()
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
                    self.notifier.update(result)
            return result

    def _detect_latest(self) -> dict | None:
        s = self.settings
        if self.buffer.empty:
            return None
        t_end = float(self.buffer["t"].max())
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
            log.warning("anomaly alert: %s score=%.3f window_end=%s", method, info["score"], result["window_end"])

    def status(self) -> dict:
        with self.lock:
            return {
                RuleBasedDetector.name: True,
                IsolationForestDetector.name: self.iforest is not None,
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
    register_dashboard(app, state, settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "UP", "version": __version__, "profile": state.profile.name, "live_detection": settings.live,
                "models": state.status(), "buffered_records": int(len(state.buffer)), "polls": state.polls,
                "telegram_notifications": state.telegram is not None}

    @app.post("/notify/test")
    def notify_test() -> dict:
        if state.telegram is None:
            raise HTTPException(status_code=400,
                                detail="Telegram is not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        try:
            state.telegram.send("✅ Test message from the log anomaly detector. Telegram alerts are working.")
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        return {"sent": True}

    @app.post("/train")
    def train(request: TrainRequest | None = None) -> dict:
        request = request or TrainRequest()
        try:
            return state.train(request.logs_dir or settings.log_dir, request.labels_path or settings.labels_path,
                               request.start, request.end)
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

    @app.get("/alerts")
    def alerts(limit: int = 50) -> dict:
        limit = max(1, min(limit, 1000))
        with state.lock:
            items = list(state.alerts)[-limit:]
        return {"count": len(items), "alerts": list(reversed(items))}

    return app


app = create_app()
