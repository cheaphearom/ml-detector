"""Reading the logs from Loki (LOG_SOURCE=loki), against a mock of Loki's query API."""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app
from detector.loki import LokiClient, LokiError, LokiTailer, entries_to_frame, load_logs_from_loki

from pos_logs import pos_lines

NS = 1_000_000_000


def ts_ns(line: str) -> int:
    return int(pd.Timestamp(json.loads(line)["@timestamp"]).value)


class MockLoki:
    """Holds lines per stream and answers /loki/api/v1/query_range like Loki (forward, limit)."""

    def __init__(self, limit: int = 5000):
        self.streams: dict[tuple, list[tuple[int, str]]] = {}
        self.queries: list[dict] = []
        self.down = False
        self.max_limit = limit
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if mock.down:
                    self.send_response(503)
                    self.end_headers()
                    return
                url = urllib.parse.urlparse(self.path)
                if url.path == "/ready":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ready")
                    return
                q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                mock.queries.append({**q, "org": self.headers.get("X-Scope-OrgID"), "auth": self.headers.get("Authorization")})
                start, end, limit = int(q["start"]), int(q["end"]), min(int(q["limit"]), mock.max_limit)
                rows = sorted((ts, line, labels) for labels, values in mock.streams.items()
                              for ts, line in values if start <= ts <= end)[:limit]
                result = {}
                for ts, line, labels in rows:
                    result.setdefault(labels, []).append([str(ts), line])
                body = json.dumps({"status": "success", "data": {"resultType": "streams", "result": [
                    {"stream": dict(labels), "values": values} for labels, values in result.items()]}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def push(self, lines, **labels):
        """Stores lines like Alloy does: the service from the JSON as a label."""
        for line in lines:
            service = json.loads(line)["service"]
            key = tuple(sorted({"job": "pos", "service": service, **labels}.items()))
            self.streams.setdefault(key, []).append((ts_ns(line), line))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def loki():
    mock = MockLoki()
    yield mock
    mock.close()


def test_query_range_returns_all_lines_in_time_order_across_streams(loki):
    lines = pos_lines(300, seed=2)
    loki.push(lines)
    entries = LokiClient(loki.url).query_range(0, 2**62)
    assert len(entries) == len(lines)
    assert [e[0] for e in entries] == sorted(e[0] for e in entries)
    assert {e[2]["service"] for e in entries} == {"pos-api", "pos-worker"}
    assert loki.queries[0]["query"] == '{job="pos"}' and loki.queries[0]["direction"] == "forward"


def test_paging_past_the_line_limit_loses_and_repeats_nothing(loki):
    lines = pos_lines(600, seed=3)
    loki.push(lines)
    client = LokiClient(loki.url, limit=100)
    entries = client.query_range(0, 2**62)
    assert len(loki.queries) > len(lines) // 100
    assert sorted(line for _, line, _ in entries) == sorted(lines)


def test_lines_with_the_same_timestamp_across_a_page_edge(loki):
    """A page ends in the middle of lines that share one nanosecond: the next page asks for that
    nanosecond again and drops the lines already read. (More lines in one nanosecond than a whole
    page, 5000 by default, is not supported: the reader moves on rather than loop.)"""
    t0 = 1_800_000_000 * NS
    lines = [json.dumps({"@timestamp": "2027-01-15T08:00:00Z", "service": "pos-api", "level": "INFO",
                         "message": f"line {i}"}) for i in range(6)]
    key = (("job", "pos"), ("service", "pos-api"))
    loki.streams[key] = [(t0 - 1, lines[0]), (t0, lines[1]), (t0, lines[2]), (t0, lines[3]),
                         (t0 + 1, lines[4]), (t0 + 1, lines[5])]
    entries = LokiClient(loki.url, limit=3).query_range(0, 2**62)
    assert len(entries) == 6
    assert sorted(line for _, line, _ in entries) == sorted(lines)


def test_entries_parse_like_log_files_and_the_service_label_fills_in(loki):
    line = json.dumps({"@timestamp": "2026-11-12T06:00:00.123Z", "level": "WARN", "message": "x event=payment_failed",
                       "event": "payment_failed", "storeId": "s1", "storeName": "Phnom Penh - Central"})
    frame = entries_to_frame([(1, line, {"service": "pos-api", "job": "pos"})])
    row = frame.iloc[0]
    assert (row["service"], row["event"], row["store_name"]) == ("pos-api", "payment_failed", "Phnom Penh - Central")


def test_headers_for_a_multi_tenant_or_protected_loki(loki):
    LokiClient(loki.url, org_id="pos", username="det", password="pw").query_range(0, 1)
    assert loki.queries[0]["org"] == "pos" and loki.queries[0]["auth"].startswith("Basic ")


def test_tailer_reads_only_new_lines_up_to_the_delay(loki):
    lines = pos_lines(300, seed=4)
    first, last = ts_ns(lines[0]) / NS, ts_ns(lines[-1]) / NS
    now = [first + 100 + 10]                                   # "now": 100 s into the log, delay 10 s
    tailer = LokiTailer(LokiClient(loki.url), delay_s=10, backfill_s=200, now=lambda: now[0])
    loki.push(lines)
    a = tailer.read_new()
    assert len(a) and a["t"].max() <= first + 100 + 1e-6
    now[0] = last + 60
    b = tailer.read_new()
    assert len(a) + len(b) == len(entries_to_frame([(0, l, {}) for l in lines]))
    assert tailer.read_new().empty                             # nothing new


def test_tailer_keeps_its_place_while_loki_is_down(loki, caplog):
    lines = pos_lines(120, seed=5)
    loki.push(lines)
    start = ts_ns(lines[0]) / NS
    now = [start + 60]
    tailer = LokiTailer(LokiClient(loki.url, timeout_s=2), delay_s=5, backfill_s=120, now=lambda: now[0])
    loki.down = True
    assert tailer.read_new().empty and "HTTP 503" in tailer.last_error
    loki.down = False
    now[0] = start + 200
    assert len(tailer.read_new()) == len(entries_to_frame([(0, l, {}) for l in lines]))   # nothing lost
    assert tailer.last_error is None


def test_unreachable_loki_is_an_error_not_a_crash():
    with pytest.raises(LokiError, match="not reachable"):
        LokiClient("http://127.0.0.1:9", timeout_s=2).query_range(0, 1)
    assert LokiClient("http://127.0.0.1:9").ready() is False


def test_training_loads_in_hour_chunks(loki):
    lines = pos_lines(7200, seed=6)
    loki.push(lines)
    start = pd.Timestamp(json.loads(lines[0])["@timestamp"])
    logs = load_logs_from_loki(LokiClient(loki.url), start, start + pd.Timedelta(hours=2))
    assert len(logs) == len(lines)
    assert len(loki.queries) >= 2


def test_detector_trains_and_detects_from_loki(loki, tmp_path):
    lines = pos_lines(3600, seed=7, fault=(3300, 3480, "WEBHOOK"))
    loki.push(lines)
    end = ts_ns(lines[-1]) / NS
    settings = Settings(profile="pos", log_source="loki", loki_url=loki.url, loki_delay_s=0,
                        model_dir=str(tmp_path / "models"), live=False, tail_from_start=True, buffer_minutes=10,
                        labels_path=str(tmp_path / "none.csv"))
    with TestClient(create_app(settings)) as c:
        state = c.app.state.detector
        state.tailer.now = lambda: end + 1                     # "now" = just after the last line
        state.tailer.cursor_ns = int((end - 600) * NS)
        start = pd.Timestamp(json.loads(lines[0])["@timestamp"]).isoformat()
        trained = c.post("/train", json={"start": start, "end": pd.Timestamp(end, unit="s", tz="UTC").isoformat()}).json()
        assert trained["pca"] is True and trained["excluded_windows"] > 0
        body = c.get("/detect").json()
        assert body["methods"]["rules"]["alert"] or body["methods"]["pca"]["alert"]
        health = c.get("/health").json()
        assert health["log_source"]["type"] == "loki" and health["log_source"]["error"] is None


def test_unknown_log_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match="LOG_SOURCE"):
        create_app(Settings(log_source="kafka", model_dir=str(tmp_path), live=False))


@pytest.mark.skipif(not os.environ.get("LOKI_TEST_URL"), reason="set LOKI_TEST_URL to a real Loki (docker compose)")
def test_against_a_real_loki():
    """Pushes lines to a real Loki and reads them back through the detector's client."""
    url = os.environ["LOKI_TEST_URL"].rstrip("/")
    run = f"t{int(time.time())}"
    now = time.time_ns()
    values = [[str(now - (50 - i) * NS), json.dumps({"@timestamp": pd.Timestamp(now - (50 - i) * NS).isoformat() + "Z",
                                                     "service": "pos-api", "level": "INFO", "message": f"line {i}"})]
              for i in range(50)]
    body = json.dumps({"streams": [{"stream": {"job": "detector-test", "run": run, "service": "pos-api"},
                                    "values": values}]}).encode()
    urllib.request.urlopen(urllib.request.Request(f"{url}/loki/api/v1/push", data=body,
                                                  headers={"Content-Type": "application/json"}), timeout=10)
    client = LokiClient(url, query=f'{{job="detector-test", run="{run}"}}', limit=20)
    for _ in range(30):
        entries = client.query_range(now - 120 * NS, now + NS)
        if len(entries) == 50:
            break
        time.sleep(1)
    assert len(entries) == 50 and [e[1] for e in entries] == [v[1] for v in values]
