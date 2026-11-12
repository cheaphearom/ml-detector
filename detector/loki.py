"""Reading the logs from Grafana Loki instead of log files (``LOG_SOURCE=loki``).

The services keep writing JSON lines; a log shipper (Grafana Alloy) pushes them to Loki, and the
detector asks Loki for them with ``/loki/api/v1/query_range``. The lines are parsed exactly like
the lines of a log file (:mod:`detector.parsing`).

Live detection (:class:`LokiTailer`): every poll asks for the lines since the previous poll, up to
``delay_s`` seconds ago, because a line can reach Loki a few seconds after it was written. Lines at
the edge of two polls are de-duplicated. Training (:func:`load_logs_from_loki`) pages through long
periods: Loki returns at most ``limit`` lines per query.

Configuration (environment variables): LOKI_URL (e.g. http://loki:3100), LOKI_QUERY (LogQL stream
selector, default ``{job="pos"}``), LOKI_DELAY_S (default 10), LOKI_ORG_ID (X-Scope-OrgID, only for a
multi-tenant Loki), LOKI_USERNAME / LOKI_PASSWORD (basic auth, e.g. a Loki behind a proxy).
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .parsing import empty_logs, iter_records, records_to_frame

log = logging.getLogger("detector.loki")

NS = 1_000_000_000


class LokiError(RuntimeError):
    pass


class LokiClient:
    def __init__(self, url: str, query: str = '{job="pos"}', org_id: str = "", username: str = "",
                 password: str = "", timeout_s: float = 15.0, limit: int = 5000):
        if not url:
            raise ValueError("LOKI_URL is required with LOG_SOURCE=loki")
        self.url = url.rstrip("/")
        self.query = query
        self.timeout_s = timeout_s
        self.limit = limit
        self.headers = {"Accept": "application/json"}
        if org_id:
            self.headers["X-Scope-OrgID"] = org_id
        if username:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {token}"

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{self.url}/ready", headers=self.headers),
                                        timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def query_range(self, start_ns: int, end_ns: int) -> list[tuple[int, str, dict]]:
        """(timestamp ns, line, stream labels) for every line in [start, end], oldest first.

        Pages through the range when Loki returns ``limit`` lines (the last timestamp of a page is
        asked again, and the lines seen twice are dropped).
        """
        out: list[tuple[int, str, dict]] = []
        seen_at_edge: set[tuple[int, str]] = set()
        cursor = start_ns
        while cursor <= end_ns:
            entries = self._query(cursor, end_ns)
            if not entries:
                break
            last_ts = entries[-1][0]
            fresh = [e for e in entries if (e[0], e[1]) not in seen_at_edge]
            out += fresh
            if len(entries) < self.limit:
                break
            if last_ts == cursor:
                cursor += 1          # a whole page in one nanosecond: move on (never loop)
                continue
            seen_at_edge = {(ts, line) for ts, line, _ in entries if ts == last_ts}
            cursor = last_ts         # same nanosecond again: more lines may share it
        return out

    def _query(self, start_ns: int, end_ns: int) -> list[tuple[int, str, dict]]:
        params = urllib.parse.urlencode({"query": self.query, "start": str(start_ns), "end": str(end_ns),
                                         "limit": str(self.limit), "direction": "forward"})
        request = urllib.request.Request(f"{self.url}/loki/api/v1/query_range?{params}", headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LokiError(f"Loki returned HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LokiError(f"Loki is not reachable at {self.url}: {getattr(e, 'reason', e)}") from None
        entries = [(int(ts), line, s.get("stream", {}))
                   for s in body.get("data", {}).get("result", []) for ts, line in s.get("values", [])]
        entries.sort(key=lambda e: e[0])
        return entries[: self.limit]


def entries_to_frame(entries: list[tuple[int, str, dict]]) -> pd.DataFrame:
    """Parses Loki lines like log-file lines; the stream's ``service`` label fills a missing field."""
    if not entries:
        return empty_logs()
    frames = []
    by_service: dict[str, list[str]] = {}
    for _, line, labels in entries:
        by_service.setdefault(labels.get("service", ""), []).append(line)
    for service, lines in by_service.items():
        frames.append(records_to_frame(iter_records(lines), default_service=service or None))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_logs()
    return pd.concat(frames, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)


def load_logs_from_loki(client: LokiClient, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """All lines between two times (training), queried an hour at a time."""
    start_ns, end_ns = int(start.value), int(end.value)
    frames = []
    step = 3600 * NS
    for chunk_start in range(start_ns, end_ns + 1, step):
        chunk_end = min(chunk_start + step - 1, end_ns)
        frames.append(entries_to_frame(client.query_range(chunk_start, chunk_end)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_logs()
    return pd.concat(frames, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)


class LokiTailer:
    """Same interface as :class:`detector.parsing.LogTailer`: ``read_new()`` returns the new lines.

    Starts ``backfill_s`` seconds ago (``tail_from_start``: the buffer is filled at once) and then
    reads up to ``delay_s`` seconds before now on every poll.
    """

    def __init__(self, client: LokiClient, delay_s: float = 10.0, backfill_s: float = 0.0,
                 now=time.time):
        self.client = client
        self.delay_s = delay_s
        self.now = now
        self.cursor_ns = int((now() - delay_s - backfill_s) * NS)
        self.last_error: str | None = None

    def read_new(self) -> pd.DataFrame:
        end_ns = int((self.now() - self.delay_s) * NS)
        if end_ns <= self.cursor_ns:
            return empty_logs()
        try:
            entries = self.client.query_range(self.cursor_ns + 1, end_ns)
        except LokiError as e:
            # keep the cursor: the lines are read once Loki answers again
            if self.last_error != str(e):
                log.warning("%s; will try again", e)
            self.last_error = str(e)
            return empty_logs()
        self.last_error = None
        self.cursor_ns = end_ns
        return entries_to_frame(entries)
