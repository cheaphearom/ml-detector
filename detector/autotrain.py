"""Automatic training: the first model once there is a day of logs, then a fresh one every week.

Training by hand (``detector.cli train``) is easy to forget, and a model trained once slowly drifts from the shop's
real "normal" (new products, a new store, busier season). :class:`AutoTrainer` decides when to train; the
detector's live loop runs it in the background, so detection goes on with the old model meanwhile.

Safety, so a bad period is never learned as normal:

* windows where the rules fire are left out (as in every training), and so are the periods the detector itself
  alerted on: :class:`IncidentLog` keeps them in ``<MODEL_DIR>/incidents.json`` (also across restarts);
* no training while an incident is open: it waits for the next day;
* if more than ``max_unusual`` (20 %) of the period was left out, the period is not "normal" enough: the old model
  stays and the owner is told;
* the first model needs at least ``MIN_TRAIN_HOURS`` (24) of logs.

Configuration (environment variables): AUTO_TRAIN (off; on in docker-compose.yml), AUTO_TRAIN_EVERY_DAYS (7),
AUTO_TRAIN_AT (04:30, in NOTIFY_TIMEZONE), AUTO_TRAIN_HOURS (24), AUTO_TRAIN_MAX_UNUSUAL (0.2).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo


def _zone(name: str) -> dt.tzinfo:
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - unknown name: UTC
        return dt.timezone.utc


class IncidentLog:
    """Periods (epoch start, end) the detector alerted on, merged, kept ``keep_days`` days, saved to a JSON file."""

    def __init__(self, path: str | None, keep_days: float = 30.0, margin_s: float = 60.0):
        self.path = path
        self.keep_s = keep_days * 86400.0
        self.margin_s = margin_s
        self.periods: list[list[float]] = []
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.periods = [[float(a), float(b)] for a, b in json.load(f)]
            except (OSError, ValueError, TypeError):
                self.periods = []

    def add(self, start: float, end: float) -> None:
        start, end = start - self.margin_s, end + self.margin_s
        if self.periods and start <= self.periods[-1][1]:
            if end <= self.periods[-1][1]:
                return
            self.periods[-1][1] = end
        else:
            self.periods.append([start, end])
        horizon = end - self.keep_s
        self.periods = [p for p in self.periods if p[1] >= horizon]
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".part"
            with open(tmp, "w") as f:
                json.dump(self.periods, f)
            os.replace(tmp, self.path)
        except OSError:
            pass   # best effort: training then only leaves out the rules' windows


@dataclass
class AutoTrainer:
    """When to train: daily at ``at`` (local time); the first model as soon as possible, then every ``every_days``."""

    every_days: float = 7.0
    at: str = "04:30"
    hours: float = 24.0
    timezone: str = "UTC"
    max_unusual: float = 0.2
    running: bool = False
    last_try_day: str | None = None
    last: dict = field(default_factory=dict)

    def _local(self, now: float) -> dt.datetime:
        return dt.datetime.fromtimestamp(now, tz=_zone(self.timezone))

    def _at(self) -> tuple[int, int]:
        try:
            h, m = (int(x) for x in self.at.split(":", 1))
            return max(0, min(23, h)), max(0, min(59, m))
        except ValueError:
            return 4, 30

    def due(self, now: float, trained_at: str | None, incident_open: bool) -> bool:
        """True when a training should start now. Records why not, for /health."""
        if self.running:
            return False
        local = self._local(now)
        today = local.date().isoformat()
        h, m = self._at()
        if (local.hour, local.minute) < (h, m) or self.last_try_day == today:
            return False
        if trained_at is not None:
            age_days = (now - dt.datetime.fromisoformat(trained_at).timestamp()) / 86400.0
            if age_days < self.every_days - 0.25:      # a few hours early is fine: same time of day each week
                return False
        self.last_try_day = today
        if incident_open:
            self.last = {"at": local.isoformat(), "ok": False, "message": "postponed: an incident is open; next try tomorrow"}
            return False
        return True

    def next_try(self, now: float, trained_at: str | None) -> str:
        """When the next training will be tried (local time), for /health."""
        local = self._local(now)
        h, m = self._at()
        # the earliest moment a training may start: now, or when the model is old enough
        earliest = local
        if trained_at is not None:
            old_enough = dt.datetime.fromisoformat(trained_at).timestamp() + (self.every_days - 0.25) * 86400.0
            earliest = max(local, self._local(old_enough))
        cand = dt.datetime.combine(earliest.date(), dt.time(h, m), tzinfo=local.tzinfo)
        if cand < earliest or self.last_try_day == cand.date().isoformat():
            cand += dt.timedelta(days=1)
        return cand.isoformat()

    def view(self, now: float, trained_at: str | None) -> dict:
        return {"every_days": self.every_days, "at": self.at, "timezone": self.timezone, "hours": self.hours,
                "max_unusual": self.max_unusual, "running": self.running, "next_try": self.next_try(now, trained_at),
                "last": self.last or None}


def trained_message(result: dict, first: bool, system: str, next_try: str) -> str:
    where = f" for the {system}" if system else ""
    left_out = int(result.get("excluded_windows", 0))
    return (f"🧠 Detector {'trained for the first time' if first else 're-trained'}{where}\n\n"
            f"Learned from the last {result.get('hours', 0):.0f} hours: {result.get('windows', 0)} windows, "
            f"{left_out} unusual ones left out.\nNext training: {next_try[:16].replace('T', ' ')}.")


def skipped_message(reason: str, system: str, next_try: str) -> str:
    where = f" for the {system}" if system else ""
    return (f"⚠️ Detector training skipped{where}\n\n{reason}\n"
            f"The current model stays. Next try: {next_try[:16].replace('T', ' ')}.")


