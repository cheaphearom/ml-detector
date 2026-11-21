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

Three more checks look at what the tills and the owner depend on beyond the API itself:

* **the public address** (:class:`HealthProbe` named ``public_url``): the same ``/health``, but through the address the
  tills use (Tailscale Funnel, Cloudflare Tunnel...). The API can be fine while the tunnel or the web app is not.
* **the server** (:class:`HostWatch`): disk usage of the folder ``disk_path`` (the server's disk) and available
  memory (``/proc/meminfo``: in a container it shows the server's memory).
* **the database backups** (:class:`BackupWatch`): the newest ``pos-*.sql.gz`` in the POS backup folder must be
  recent and not suspiciously small. Armed once a first backup file is seen.

Configuration (environment variables): POS_HEALTH_URL (empty = no probe), HEALTH_CHECK_INTERVAL_S (30),
HEALTH_FAIL_AFTER (2), HEARTBEAT_TIMEOUT_S (120, 0 = off), HEARTBEAT_EVENT (heartbeat), PUBLIC_HEALTH_URL (empty = off),
PUBLIC_CHECK_INTERVAL_S (60), HOST_CHECKS (off; on in docker-compose.yml), DISK_CHECK_PATH, DISK_ALERT_PCT (85),
MEMORY_ALERT_PCT (10), BACKUP_DIR (empty = off), BACKUP_MAX_AGE_H (26), BACKUP_MIN_KB (1).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
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
    name: str = "pos_api"
    state: CheckState = field(init=False)
    last_ms: float | None = None
    _next: float = 0.0

    def __post_init__(self):
        self.state = CheckState(self.name, self.fail_after)

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


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def read_meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    """MemTotal, MemAvailable, SwapTotal, SwapFree ... in bytes."""
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1)
    return out


@dataclass
class HostWatch:
    """The server's disk (the file system holding ``disk_path``) and memory, every ``interval_s`` seconds.

    Disk: "down" when more than ``disk_alert_pct`` % is used (2 checks in a row). Memory: "down" when less than
    ``memory_alert_pct`` % is available for ``memory_fail_after`` checks in a row (5 minutes by default), so a short
    peak (a build, a training) does not page anyone.
    """

    disk_path: str = "/models"
    disk_alert_pct: float = 85.0
    memory_alert_pct: float = 10.0
    interval_s: float = 60.0
    memory_fail_after: int = 5
    meminfo_path: str = "/proc/meminfo"
    disk: CheckState = field(init=False)
    memory: CheckState = field(init=False)
    last: dict = field(default_factory=dict)
    _next: float = 0.0

    def __post_init__(self):
        self.disk = CheckState("disk", fail_after=2)
        self.memory = CheckState("memory", fail_after=self.memory_fail_after)

    def due(self, now: float) -> bool:
        return now >= self._next

    def run(self, now: float) -> list[tuple[str, str, str]]:
        """Checks once; returns [(kind, transition, detail)] for the checks that changed."""
        self._next = now + self.interval_s
        changes: list[tuple[str, str, str]] = []
        if self.disk_alert_pct > 0:
            try:
                st = os.statvfs(self.disk_path)
                total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
                pct = round((total - free) / total * 100.0, 1) if total else 0.0
                detail = f"disk {pct:.0f}% full ({_size(free)} free of {_size(total)})"
                self.last.update(disk_used_pct=pct, disk_free_bytes=free, disk_total_bytes=total)
                change = self.disk.update(pct <= self.disk_alert_pct, now, detail)
            except OSError as e:
                change = None
                self.disk.last_detail = f"cannot read the disk usage of {self.disk_path}: {e}"
            if change:
                changes.append(("disk", change, self.disk.last_detail))
        if self.memory_alert_pct > 0:
            try:
                m = read_meminfo(self.meminfo_path)
                total, avail = m.get("MemTotal", 0), m.get("MemAvailable", m.get("MemFree", 0))
                swap = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
                pct = round(avail / total * 100.0, 1) if total else 100.0
                detail = (f"memory {pct:.0f}% available ({_size(avail)} of {_size(total)})"
                          + (f", swap used {_size(swap)}" if swap > 0 else ""))
                self.last.update(memory_available_pct=pct, memory_available_bytes=avail, swap_used_bytes=swap)
                change = self.memory.update(pct >= self.memory_alert_pct, now, detail)
            except (OSError, ValueError) as e:
                change = None
                self.memory.last_detail = f"cannot read the memory: {e}"
            if change:
                changes.append(("memory", change, self.memory.last_detail))
        return changes

    def view(self) -> dict:
        return {"disk": {**self.disk.view(), "path": self.disk_path, "alert_pct": self.disk_alert_pct,
                         "used_pct": self.last.get("disk_used_pct")} if self.disk_alert_pct > 0 else None,
                "memory": {**self.memory.view(), "alert_pct": self.memory_alert_pct,
                           "available_pct": self.last.get("memory_available_pct")} if self.memory_alert_pct > 0 else None}


@dataclass
class BackupWatch:
    """The newest database backup (``pattern`` in ``directory``) must be younger than ``max_age_h`` hours,
    at least ``min_bytes`` big, and not less than half the size of the one before (a failed or cut dump).

    Armed once a first backup file is seen, so a server without backups (or a wrong folder) never raises it;
    if every file then disappears, that is a problem.
    """

    directory: str
    max_age_h: float = 26.0
    min_bytes: int = 1024
    pattern: str = "pos-*.sql.gz"
    interval_s: float = 600.0
    state: CheckState = field(init=False)
    armed: bool = False
    newest: dict | None = None
    _next: float = 0.0

    def __post_init__(self):
        self.state = CheckState("backup", fail_after=1)

    def due(self, now: float) -> bool:
        return now >= self._next

    def run(self, now: float) -> tuple[str | None, str]:
        """Checks once; returns (transition, detail)."""
        self._next = now + self.interval_s
        files = []
        for path in glob.glob(os.path.join(self.directory, self.pattern)):
            try:
                st = os.stat(path)
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, os.path.basename(path)))
        files.sort(reverse=True)
        if not files:
            if not self.armed:
                self.state.last_detail = f"waiting for a first backup in {self.directory}"
                return None, self.state.last_detail
            return self.state.update(False, now, f"no backup file left in {self.directory}"), self.state.last_detail
        self.armed = True
        mtime, size, name = files[0]
        age_h = (now - mtime) / 3600.0
        self.newest = {"file": name, "at": _clock(mtime), "bytes": size, "age_h": round(age_h, 1), "count": len(files)}
        problems = []
        if age_h > self.max_age_h:
            problems.append(f"the last backup is {age_h:.0f} hours old ({name})")
        if size < self.min_bytes:
            problems.append(f"the last backup ({name}) is only {_size(size)}: the database dump probably failed")
        elif len(files) > 1 and files[1][1] >= 4 * self.min_bytes and size < files[1][1] / 2:
            problems.append(f"the last backup ({name}, {_size(size)}) is less than half the size of the one before "
                            f"({_size(files[1][1])}): check it")
        detail = "; ".join(problems) if problems else f"last backup {name}, {_size(size)}, {age_h:.0f} h ago"
        return self.state.update(not problems, now, detail), detail

    def view(self) -> dict:
        return {**self.state.view(), "armed": self.armed, "directory": self.directory, "max_age_h": self.max_age_h,
                "newest": self.newest}


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
    now = clock(time.time())
    minutes = f" (about {(time.time() - since) / 60:.0f} min)" if since else ""
    if kind == "public_url":
        if transition == "down":
            return (f"🚨 POS not reachable from outside{where}\nTime: {now}\n\n"
                    f"The public address did not answer twice in a row: {detail or 'no answer'}.\n"
                    f"Tills and phones cannot reach the POS. If no \"API not responding\" message came, the API itself "
                    f"is fine: check the tunnel (Tailscale Funnel / Cloudflare) and the web app.")
        return f"✅ POS reachable from outside again{where}{minutes}\nTime: {now}"
    if kind == "disk":
        if transition == "down":
            return (f"🚨 Server disk almost full{where}\nTime: {now}\n\n{detail}.\n"
                    f"When it is full the database and the POS stop. Free space on the server (removes only old build "
                    f"leftovers): docker builder prune -f && docker image prune -f")
        return f"✅ Server disk OK again{where}: {detail}\nTime: {now}"
    if kind == "memory":
        if transition == "down":
            return (f"🚨 Server memory almost full{where}\nTime: {now}\n\n{detail}, for several minutes.\n"
                    f"The POS may become slow, or a container may be stopped.")
        return f"✅ Server memory OK again{where}: {detail}\nTime: {now}"
    if kind == "backup":
        if transition == "down":
            return (f"🚨 Database backup problem{where}\nTime: {now}\n\n{detail}.\n"
                    f"Check on the server: cd ~/pos-backend/deploy && docker compose logs backup --tail 20")
        return f"✅ Database backups OK again{where}: {detail}\nTime: {now}"
    if transition == "down":
        return (f"🚨 No logs from the API{where}\nTime: {now}\n\n{detail}.\n"
                f"Until logs arrive again, the anomaly detection is blind.")
    return f"✅ Logs arriving again{where}\nTime: {now}"
