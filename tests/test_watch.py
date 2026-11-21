"""System checks (API probe, heartbeat) and the detector's self-monitoring lines."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import detector.api as api_module
from detector.api import Settings, create_app
from detector.selfmon import SelfMonitor
from detector.watch import CheckState, HealthProbe, HeartbeatWatch

from pos_logs import pos_lines


class FakeServer:
    """GET answers with ``status`` / ``body``; POST /loki/api/v1/push records the pushed streams."""

    def __init__(self):
        self.status, self.body = 200, {"status": "ok", "database": "ok", "redis": "ok"}
        self.pushes: list[dict] = []
        self.down = False
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                data = json.dumps(me.body).encode()
                self.send_response(me.status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if me.down:
                    self.send_response(503)
                    self.end_headers()
                    return
                me.pushes.append(body)
                self.send_response(204)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def lines(self):
        return [json.loads(v[1]) for p in self.pushes for s in p["streams"] for v in s["values"]]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def server():
    s = FakeServer()
    yield s
    s.close()


# ── CheckState ─────────────────────────────────────────────────────────────────────────
def test_one_down_and_one_up_message():
    c = CheckState("x", fail_after=2)
    assert [c.update(ok, t) for t, ok in enumerate([True, False, True, False, False, False, False, True, True])] == \
        [None, None, None, None, "down", None, None, "up", None]


# ── API probe ──────────────────────────────────────────────────────────────────────────
def test_probe_healthy_unhealthy_and_unreachable(server):
    p = HealthProbe(server.url + "/health", interval_s=30, fail_after=2)
    assert p.run(0) == (None, {"ok": True, "status": 200, "ms": p.last_ms, "detail": ""})
    assert not p.due(10) and p.due(30)
    server.status, server.body = 503, {"status": "fail", "database": "fail", "redis": "ok"}
    change, res = p.run(30)
    assert change is None and res["detail"] == "HTTP 503 (database: fail)"
    change, _ = p.run(60)
    assert change == "down" and p.state.view()["status"] == "down"
    server.status, server.body = 200, {"status": "ok"}
    assert p.run(90)[0] == "up"

    gone = HealthProbe("http://127.0.0.1:9/health", timeout_s=2, fail_after=1)
    change, res = gone.run(0)
    assert change == "down" and res["status"] is None and res["detail"].startswith("no answer")


# ── heartbeat ──────────────────────────────────────────────────────────────────────────
def test_heartbeat_arms_on_the_first_beat_and_raises_after_the_timeout():
    w = HeartbeatWatch(timeout_s=120)
    assert w.check(1000, None) == (None, "")                  # never saw one: not armed
    w.saw([995.0], now=1000)
    assert w.check(1100, None)[0] is None
    change, reason = w.check(1121, None)
    assert change == "down" and "no heartbeat for 2 min" in reason
    assert w.check(1200, None)[0] is None                      # still down: no second message
    w.saw([1199.0], now=1201)
    assert w.check(1202, None)[0] == "up"


def test_heartbeat_names_a_loki_read_error():
    w = HeartbeatWatch(timeout_s=60)
    w.saw([1.0], now=0)
    change, reason = w.check(61, "Loki is not reachable at http://loki:3100: refused")
    assert change == "down" and reason.startswith("the detector cannot read the logs: Loki is not reachable")


def test_heartbeat_off():
    w = HeartbeatWatch(timeout_s=0)
    w.saw([1.0], now=0)
    assert w.check(10_000, None) == (None, "")


# ── self-monitoring ────────────────────────────────────────────────────────────────────
def test_self_monitor_pushes_json_lines_with_its_own_label(server):
    m = SelfMonitor(server.url, job="ml-detector")
    m.event("poll", lines_read=12, score_rules=0.1)
    m.event("alert", level="WARN", method="pca")
    assert m.flush() and m.sent == 2
    streams = server.pushes[0]["streams"]
    assert {tuple(sorted(s["stream"].items())) for s in streams} == {(("job", "ml-detector"), ("level", "INFO")),
                                                                    (("job", "ml-detector"), ("level", "WARN"))}
    assert {l["event"] for l in server.lines()} == {"poll", "alert"}
    server.down = True
    m.event("poll")
    assert m.flush() is False and m.dropped == 1 and "could not write to Loki" in m.last_error


# ── the live loop ──────────────────────────────────────────────────────────────────────
def heartbeat_line(t_iso):
    return json.dumps({"@timestamp": t_iso, "level": "INFO", "service": "pos-api", "logger_name": "Heartbeat",
                       "event": "heartbeat", "message": "Heartbeat event=heartbeat"})


def test_live_loop_runs_both_checks_and_writes_its_activity(tmp_path, server, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    lines = pos_lines(300, seed=31)
    beats = [heartbeat_line(f"2026-11-12T06:0{m}:{s:02d}.000Z") for m in range(5) for s in (0, 30)]
    (logs / "pos-api.log").write_text("\n".join(sorted(lines + beats, key=lambda l: json.loads(l)["@timestamp"])) + "\n")
    sent = []
    settings = Settings(profile="pos", log_dir=str(logs), model_dir=str(tmp_path / "m"), live=False,
                        tail_from_start=True, labels_path=str(tmp_path / "none.csv"),
                        pos_health_url=server.url + "/health", health_check_interval_s=0, health_fail_after=2,
                        heartbeat_timeout_s=120, self_monitoring="on", loki_url=server.url)
    clock = [1_800_000_000.0]
    monkeypatch.setattr(api_module.time, "time", lambda: clock[0])
    with TestClient(create_app(settings, send=sent.append)) as c:
        state = c.app.state.detector
        c.get("/detect")
        # heartbeats were seen, and kept out of the features (no heartbeat event in the buffer)
        assert state.heartbeat.last_heartbeat is not None
        assert not (state.buffer["event"] == "heartbeat").any()
        assert c.get("/health").json()["checks"]["pos_api"]["status"] == "ok"

        # the API goes down: two failed probes -> one message, one alert row
        server.status, server.body = 503, {"status": "fail", "database": "fail", "redis": "ok"}
        clock[0] += 10
        c.get("/detect")
        clock[0] += 10
        c.get("/detect")
        clock[0] += 10
        c.get("/detect")
        down = [m for m in sent if "API not responding" in m]
        assert len(down) == 1 and "HTTP 503 (database: fail)" in down[0] and "in the POS" in down[0]
        alerts = c.get("/alerts").json()["alerts"]
        assert alerts[0]["method"] == "pos_api_check" and "database: fail" in alerts[0]["explanation"]["suspected_problem"]

        # ... and comes back
        server.status, server.body = 200, {"status": "ok"}
        clock[0] += 10
        c.get("/detect")
        assert sum("API responding again" in m for m in sent) == 1

        # no new heartbeat for 2 minutes -> "No logs"
        clock[0] += 200
        c.get("/detect")
        no_logs = [m for m in sent if "No logs from the API" in m]
        assert len(no_logs) == 1 and "no heartbeat for" in no_logs[0]
        health = c.get("/health").json()
        assert health["checks"]["heartbeat"]["status"] == "down" and health["checks"]["heartbeat"]["armed"] is True

    events = [l["event"] for l in server.lines()]
    assert {"poll", "pos_api_check", "check_down", "check_up"} <= set(events)
    poll = next(l for l in server.lines() if l["event"] == "poll" and "score_rules" in l)
    assert {"lines_read", "buffered", "lag_s", "threshold_rules", "alert_rules", "heartbeat_age_s", "pos_api_ms"} <= set(poll)


def test_no_probe_and_no_self_monitoring_by_default(tmp_path):
    with TestClient(create_app(Settings(profile="pos", log_dir=str(tmp_path), model_dir=str(tmp_path / "m"),
                                        live=False))) as c:
        health = c.get("/health").json()
        assert health["checks"]["pos_api"] is None and health["self_monitoring"] is None
        assert health["checks"]["heartbeat"]["armed"] is False


# ── heartbeats keep the window clock moving when nobody uses the system ───────────────────
def _beats(start_s, end_s, every=30):
    import pandas as pd
    t0 = pd.Timestamp("2026-11-12T06:00:00Z")
    return [heartbeat_line((t0 + pd.Timedelta(seconds=s)).isoformat().replace("+00:00", "Z"))
            for s in range(start_s, end_s, every)]


def test_quiet_period_after_an_incident_still_resolves(tmp_path):
    """Webhooks fail, then the shop is closed: only heartbeats arrive. The incident must still be resolved."""
    logs = tmp_path / "logs"
    logs.mkdir()
    busy = pos_lines(300, seed=33, fault=(200, 300, "WEBHOOK"))          # 5 busy minutes, failing at the end
    quiet = _beats(300, 600)                                             # 5 quiet minutes: heartbeats only
    f = logs / "pos-api.log"
    f.write_text("\n".join(sorted(busy + _beats(0, 300), key=lambda l: json.loads(l)["@timestamp"])) + "\n")
    sent = []
    settings = Settings(profile="pos", log_dir=str(logs), model_dir=str(tmp_path / "m"), live=False,
                        tail_from_start=True, labels_path=str(tmp_path / "none.csv"), notify_resolve_after_s=60)
    with TestClient(create_app(settings, send=sent.append)) as c:
        body = c.get("/detect").json()
        assert body["methods"]["rules"]["alert"] and len(sent) == 1 and sent[0].startswith("🚨")
        with f.open("a") as fh:
            fh.write("\n".join(quiet) + "\n")
        body = c.get("/detect").json()
        assert body["window_end"].startswith("2026-11-12T06:09")         # the clock followed the heartbeats (06:05 -> 06:09)
        assert body["features"]["pos-api__http_count"] == 0              # an empty window, not old traffic
        assert len(sent) == 2 and sent[1].startswith("✅ Resolved in the POS")


def test_training_leaves_heartbeats_out(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    lines = pos_lines(1800, seed=34)
    (logs / "pos-api.log").write_text(
        "\n".join(sorted(lines + _beats(0, 1800), key=lambda l: json.loads(l)["@timestamp"])) + "\n")
    with TestClient(create_app(Settings(profile="pos", log_dir=str(logs), model_dir=str(tmp_path / "m"), live=False,
                                        labels_path=str(tmp_path / "none.csv")))) as c:
        assert c.post("/train", json={}).json()["log_records"] == len(lines)


# ── the public address, the server (disk, memory) and the database backups ─────────────────
from detector.watch import BackupWatch, HostWatch, message  # noqa: E402


def test_public_address_probe_has_its_own_name_and_messages(server):
    probe = HealthProbe(server.url + "/health", interval_s=0, fail_after=2, name="public_url")
    assert probe.run(0)[0] is None and probe.state.name == "public_url"
    server.status, server.body = 502, {"error": "tunnel"}
    assert probe.run(10)[0] is None
    change, res = probe.run(20)
    assert change == "down" and res["detail"] == "HTTP 502"
    text = message("public_url", "down", "POS", res["detail"], 20, str)
    assert "POS not reachable from outside" in text and "tunnel" in text
    server.status, server.body = 200, {"status": "ok"}
    assert probe.run(30)[0] == "up"
    assert "reachable from outside again" in message("public_url", "up", "POS", "", 20, str)


def _meminfo(tmp_path, total_kb, avail_kb, swap_total_kb=0, swap_free_kb=0):
    p = tmp_path / "meminfo"
    p.write_text(f"MemTotal: {total_kb} kB\nMemFree: 100 kB\nMemAvailable: {avail_kb} kB\n"
                 f"SwapTotal: {swap_total_kb} kB\nSwapFree: {swap_free_kb} kB\n")
    return str(p)


def test_disk_alerts_above_the_limit_after_two_checks(tmp_path):
    w = HostWatch(str(tmp_path), disk_alert_pct=99.9999, memory_alert_pct=0, interval_s=0)
    assert w.run(0) == [] and w.disk.last_detail.startswith("disk ")
    w.disk_alert_pct = 0.0001  # any real disk is fuller than this
    assert w.run(60) == []
    (kind, change, detail), = w.run(120)
    assert (kind, change) == ("disk", "down") and "% full" in detail and "free of" in detail
    assert "docker builder prune" in message("disk", "down", "POS", detail, 120, str)
    w.disk_alert_pct = 99.9999
    assert w.run(180)[0][:2] == ("disk", "up")
    assert w.view()["memory"] is None and w.view()["disk"]["status"] == "ok"


def test_memory_alerts_only_when_low_for_five_checks(tmp_path):
    w = HostWatch(str(tmp_path), disk_alert_pct=0, memory_alert_pct=10, interval_s=0,
                  meminfo_path=_meminfo(tmp_path, 4_000_000, 2_000_000))
    assert w.run(0) == [] and "50% available" in w.memory.last_detail
    w.meminfo_path = _meminfo(tmp_path, 4_000_000, 200_000, 2_000_000, 500_000)   # 5% available, 1.4 GB swap used
    assert all(w.run(t) == [] for t in (60, 120, 180, 240))                         # a short peak: nothing
    (kind, change, detail), = w.run(300)
    assert (kind, change) == ("memory", "down") and "5% available" in detail and "swap used" in detail
    w.meminfo_path = _meminfo(tmp_path, 4_000_000, 3_000_000)
    assert w.run(360)[0][:2] == ("memory", "up")


def _backup(dir_, name, size, age_h, now):
    import os
    p = dir_ / name
    p.write_bytes(b"x" * size)
    os.utime(p, (now - age_h * 3600, now - age_h * 3600))


def test_backups_arm_on_the_first_file_then_check_age_and_size(tmp_path):
    now = 1_800_000_000.0
    w = BackupWatch(str(tmp_path), max_age_h=26, min_bytes=1024, interval_s=0)
    assert w.run(now) == (None, f"waiting for a first backup in {tmp_path}") and not w.armed
    _backup(tmp_path, "pos-20260101-033000.sql.gz", 80_000, 30, now)
    _backup(tmp_path, "pos-20260102-033000.sql.gz", 82_000, 6, now)
    change, detail = w.run(now)
    assert change is None and w.armed and "6 h ago" in detail and w.view()["newest"]["count"] == 2

    # too old: the daily backup did not run
    change, detail = w.run(now + 24 * 3600)
    assert change == "down" and "30 hours old" in detail
    assert "docker compose logs backup" in message("backup", "down", "POS", detail, now, str)

    # a new one, but almost empty: the dump failed
    _backup(tmp_path, "pos-20260103-033000.sql.gz", 20, 0, now + 24 * 3600)
    change, detail = w.run(now + 24 * 3600 + 1)
    assert change is None and "dump probably failed" in detail        # still down, one message only

    # a good one: back
    _backup(tmp_path, "pos-20260104-033000.sql.gz", 83_000, 0, now + 48 * 3600)
    assert w.run(now + 48 * 3600)[0] == "up"

    # much smaller than the one before
    _backup(tmp_path, "pos-20260105-033000.sql.gz", 30_000, 0, now + 72 * 3600)
    change, detail = w.run(now + 72 * 3600)
    assert change == "down" and "less than half the size" in detail


def test_live_loop_runs_the_new_checks_and_shows_them(tmp_path, server, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pos-api.log").write_text("\n".join(pos_lines(50, seed=5)) + "\n")
    backups = tmp_path / "backups"
    backups.mkdir()
    clock = [1_800_000_000.0]
    _backup(backups, "pos-20260101-033000.sql.gz", 50_000, 40, clock[0])          # 40 h old
    sent = []
    settings = Settings(profile="pos", log_dir=str(logs), model_dir=str(tmp_path / "m"), live=False,
                        tail_from_start=True, labels_path=str(tmp_path / "none.csv"),
                        public_health_url=server.url + "/health", public_check_interval_s=0,
                        host_checks=True, disk_check_path=str(tmp_path), disk_alert_pct=99.9999, memory_alert_pct=0.0001,
                        host_check_interval_s=0, backup_dir=str(backups), backup_check_interval_s=0)
    monkeypatch.setattr(api_module.time, "time", lambda: clock[0])
    with TestClient(create_app(settings, send=sent.append)) as c:
        c.get("/detect")
        assert any("Database backup problem" in m and "40 hours old" in m for m in sent)
        server.status = 502
        for _ in range(2):
            clock[0] += 60
            c.get("/detect")
        assert sum("POS not reachable from outside" in m for m in sent) == 1
        checks = c.get("/health").json()["checks"]
        assert checks["public_url"]["status"] == "down" and checks["backup"]["status"] == "down"
        assert checks["disk"]["status"] == "ok" and checks["memory"]["status"] == "ok"
        methods = {a["method"] for a in c.get("/alerts").json()["alerts"]}
        assert {"public_url_check", "backup_check"} <= methods


def test_new_checks_are_off_by_default(tmp_path):
    with TestClient(create_app(Settings(profile="pos", log_dir=str(tmp_path), model_dir=str(tmp_path / "m"),
                                        live=False))) as c:
        checks = c.get("/health").json()["checks"]
        assert checks["public_url"] is None and checks["disk"] is None and checks["memory"] is None and checks["backup"] is None
