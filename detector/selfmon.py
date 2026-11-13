"""The detector's own activity, written to Loki for the "ML detector" Grafana dashboard.

Every poll adds one ``event=window_scored`` line (score, threshold and alert of every method, lines
read, how far behind real time the window is); alerts, incidents, Telegram sends and the system
checks add their own events. Lines go to Loki with the labels ``job="ml-detector"`` and ``level``,
never mixed with the watched system's logs (``job="pos"``).

Pushing is best effort: when Loki does not answer, the lines are dropped (a WARN is logged once) and
the detection itself is not affected.

Configuration: SELF_MONITORING (default on with LOG_SOURCE=loki), SELF_MONITORING_JOB (ml-detector).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger("detector.selfmon")


class SelfMonitor:
    def __init__(self, loki_url: str, job: str = "ml-detector", headers: dict | None = None,
                 timeout_s: float = 3.0, max_pending: int = 500):
        self.url = loki_url.rstrip("/") + "/loki/api/v1/push"
        self.job = job
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self.pending: list[tuple[int, str, str]] = []
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None

    def event(self, event: str, level: str = "INFO", **fields) -> None:
        """Queues one JSON line; ``flush`` sends the queue."""
        now = time.time_ns()
        record = {"@timestamp": dt.datetime.fromtimestamp(now / 1e9, tz=dt.timezone.utc)
                  .isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                  "level": level, "service": "ml-detector", "event": event, **fields}
        self.pending.append((now, level, json.dumps(record, default=str, separators=(",", ":"))))
        if len(self.pending) > self.max_pending:
            self.dropped += len(self.pending) - self.max_pending
            self.pending = self.pending[-self.max_pending:]

    def flush(self) -> bool:
        if not self.pending:
            return True
        streams: dict[str, list[list[str]]] = {}
        for ts, level, line in self.pending:
            streams.setdefault(level, []).append([str(ts), line])
        body = json.dumps({"streams": [{"stream": {"job": self.job, "level": level}, "values": values}
                                       for level, values in streams.items()]}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(self.url, data=body, headers=self.headers, method="POST"),
                                   timeout=self.timeout_s).close()
        except (urllib.error.URLError, OSError) as e:
            error = f"could not write to Loki: {getattr(e, 'reason', e)}"
            if self.last_error != error:
                log.warning("self-monitoring: %s (lines dropped)", error)
            self.last_error = error
            self.dropped += len(self.pending)
            self.pending = []
            return False
        self.sent += len(self.pending)
        self.pending = []
        self.last_error = None
        return True

    def view(self) -> dict:
        return {"job": self.job, "sent": self.sent, "dropped": self.dropped, "last_error": self.last_error}
