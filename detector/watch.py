"""Checks that do not depend on the logs' content: is the system there at all?

The anomaly detection only sees what the logs say. When the watched API is down, or its logs stop
arriving, there is nothing to score and the detector would stay silent. Two checks close that gap:

* :class:`HealthProbe`, **the API probe**: every ``interval_s`` seconds an HTTP GET to the system's health
  endpoint (POS: ``http://api:3000/health``). HTTP 200 is healthy; anything else (503 when the
  database or Redis is down, a timeout, no answer) is a failure.
* :class:`HeartbeatWatch`, **the heartbeat**: the API writes an ``event=heartbeat`` line every 30 s. When
  none has arrived for ``timeout_s`` (wall-clock time), the logs have stopped: the API is down, or
  the log shipping (Alloy, Loki) is broken. The watch only arms itself once it has seen a first
  heartbeat, so a system without heartbeats never raises it.

:class:`CheckState` turns a stream of ok / failing results into exactly one "down" and one "back"
message, after ``fail_after`` failures in a row.

Configuration (environment variables): POS_HEALTH_URL (empty = no probe), HEALTH_CHECK_INTERVAL_S (30),
HEALTH_FAIL_AFTER (2), HEARTBEAT_TIMEOUT_S (120, 0 = off), HEARTBEAT_EVENT (heartbeat).
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field


def _clock(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat()


@dataclass
class CheckState:
    """Ok / failing history of one check, with "down" after ``fail_after`` failures in a row."""

    name: str
    fail_after: int = 2
    failures: int = 0
    down: bool = False
    down_since: float | None = None
    last_ok: float | None = None
    last_check: float | None = None
    last_detail: str = ""
    events: int = 0

    def update(self, ok: bool, now: float, detail: str = "") -> str | None:
        """Feeds one result; returns 'down', 'up' or None (no change worth a message)."""
        self.last_check = now
        self.last_detail = detail
        if ok:
            self.failures = 0
            self.last_ok = now
            if self.down:
                self.down = False
                self.events += 1
                return "up"
            return None
        self.failures += 1
        if not self.down and self.failures >= self.fail_after:
            self.down = True
            self.down_since = now
            self.events += 1
            return "down"
        return None

    def view(self) -> dict:
        return {"status": "down" if self.down else ("ok" if self.last_ok is not None else "unknown"),
                "failures_in_a_row": self.failures, "detail": self.last_detail,
                "down_since": _clock(self.down_since) if self.down and self.down_since else None,
                "last_ok": _clock(self.last_ok) if self.last_ok else None,
                "last_check": _clock(self.last_check) if self.last_check else None}


@dataclass
class HealthProbe:
    """GET ``url`` every ``interval_s`` seconds; 200 = healthy."""

    url: str
    interval_s: float = 30.0
    timeout_s: float = 5.0
    fail_after: int = 2
    state: CheckState = field(init=False)
    last_ms: float | None = None
    _next: float = 0.0

    def __post_init__(self):
        self.state = CheckState("pos_api", self.fail_after)

    def due(self, now: float) -> bool:
        return now >= self._next

    def run(self, now: float) -> tuple[str | None, dict]:
        """Probes once; returns (transition, result) where result has ok, status, ms, detail."""
        self._next = now + self.interval_s
        start = time.perf_counter()
        status, detail = None, ""
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url, headers={"Accept": "application/json"}),
                                        timeout=self.timeout_s) as r:
                status = r.status
                body = r.read(2000).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read(2000).decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError) as e:
            body = ""
            detail = f"no answer: {getattr(e, 'reason', e)}"
        ms = round((time.perf_counter() - start) * 1000.0, 1)
        self.last_ms = ms if status is not None else None
        ok = status == 200
        if status is not None and not ok:
            detail = f"HTTP {status}" + _failed_parts(body)
        transition = self.state.update(ok, now, detail)
        return transition, {"ok": ok, "status": status, "ms": self.last_ms, "detail": detail}


def _failed_parts(body: str) -> str:
    """'database: fail' from the POS /health answer, if it says which part failed."""
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    parts = [k for k, v in data.items() if k != "status" and isinstance(v, str) and v == "fail"]
    return f" ({', '.join(f'{p}: fail' for p in parts)})" if parts else ""


@dataclass
class HeartbeatWatch:
    """Raises when no heartbeat line has arrived for ``timeout_s`` seconds (after the first one)."""

    timeout_s: float = 120.0
    state: CheckState = field(init=False)
    last_heartbeat: float | None = None      # time of the newest heartbeat line (log time)
    last_seen: float | None = None           # when the detector received it (wall clock)
    started: float = field(default_factory=time.time)

    def __post_init__(self):
        self.state = CheckState("heartbeat", fail_after=1)

    def saw(self, heartbeat_times, now: float) -> None:
        if len(heartbeat_times):
            newest = float(max(heartbeat_times))
            if self.last_heartbeat is None or newest > self.last_heartbeat:
                self.last_heartbeat = newest
                self.last_seen = now

    def check(self, now: float, read_error: str | None) -> tuple[str | None, str]:
        """Returns (transition, reason). Not armed until a first heartbeat was seen."""
        if self.timeout_s <= 0 or self.last_seen is None:
            return None, ""
        silent = now - self.last_seen
        if silent < self.timeout_s:
            return self.state.update(True, now, ""), ""
        reason = (f"the detector cannot read the logs: {read_error}" if read_error
                  else f"no heartbeat for {silent / 60:.0f} min: the API is down, or its logs do not reach Loki")
        return self.state.update(False, now, reason), reason

    def view(self, now: float) -> dict:
        v = self.state.view()
        v.update(armed=self.last_seen is not None,
                 last_heartbeat=_clock(self.last_heartbeat) if self.last_heartbeat else None,
                 seconds_since=round(now - self.last_seen, 1) if self.last_seen else None,
                 timeout_s=self.timeout_s)
        return v


def message(kind: str, transition: str, system: str, detail: str, since: float | None,
            clock: Callable[[float], str]) -> str:
    """Telegram text for a check that went down or came back."""
    where = f" in the {system}" if system else ""
    if kind == "pos_api":
        if transition == "down":
            return (f"🚨 API not responding{where}\nTime: {clock(time.time())}\n\n"
                    f"The health check failed twice in a row: {detail or 'no answer'}.\n"
                    f"Tills and the web app may not work. The anomaly detection cannot see a stopped API.")
        minutes = f" (down about {(time.time() - since) / 60:.0f} min)" if since else ""
        return f"✅ API responding again{where}{minutes}\nTime: {clock(time.time())}"
    if transition == "down":
        return (f"🚨 No logs from the API{where}\nTime: {clock(time.time())}\n\n{detail}.\n"
                f"Until logs arrive again, the anomaly detection is blind.")
    return f"✅ Logs arriving again{where}\nTime: {clock(time.time())}"
