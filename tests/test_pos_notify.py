"""POS alert wording and Telegram delivery against a mock Telegram Bot API (a local HTTP server)."""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app
from detector.explain import default_baseline, explain, fit_baseline, main_store
from detector.features import build_windows
from detector.notify import TelegramError, TelegramSender, format_detected
from detector.parsing import iter_records, records_to_frame
from detector.profiles import get_profile

from pos_logs import pos_lines, write_pos_logs

POS = get_profile("pos")


# ── mock Telegram ──────────────────────────────────────────────────────────────────────
class MockTelegram:
    """Answers POST /bot<token>/sendMessage. ``plan`` = list of HTTP statuses for the next requests
    (then 200); a status of 0 closes the connection without an answer."""

    def __init__(self):
        self.messages: list[dict] = []
        self.requests = 0
        self.plan: list[int] = []
        self.retry_after: int | None = None
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                mock.requests += 1
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status = mock.plan.pop(0) if mock.plan else 200
                if status == 0:
                    self.close_connection = True
                    self.connection.close()
                    return
                if status == 200:
                    mock.messages.append({"path": self.path, **body})
                    payload = {"ok": True, "result": {"message_id": len(mock.messages)}}
                elif status == 429:
                    payload = {"ok": False, "error_code": 429, "description": "Too Many Requests",
                               "parameters": {"retry_after": mock.retry_after or 1}}
                else:
                    payload = {"ok": False, "error_code": status, "description": {
                        400: "Bad Request: chat not found", 401: "Unauthorized"}.get(status, "Internal Server Error")}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def telegram():
    mock = MockTelegram()
    yield mock
    mock.close()


def sender(mock_url, **kw):
    waits = []
    s = TelegramSender("123:ABC", "-1001", api_url=mock_url, timeout_s=3, sleep=waits.append, **kw)
    return s, waits


# ── delivery ───────────────────────────────────────────────────────────────────────────
def test_message_reaches_the_chat(telegram):
    s, waits = sender(telegram.url)
    s.send_async("🚨 hello")
    assert s.flush()
    assert telegram.messages == [{"path": "/bot123:ABC/sendMessage", "chat_id": "-1001", "text": "🚨 hello",
                                  "disable_web_page_preview": True}]
    assert s.sent == 1 and waits == []


def test_telegram_down_is_retried_later_and_logged(telegram, caplog):
    telegram.plan = [503, 0, 502]            # error, connection dropped, error, then OK
    s, waits = sender(telegram.url)
    with caplog.at_level(logging.WARNING, logger="detector.notify"):
        s.send_async("first")
        s.send_async("second")
        assert s.flush()
    assert [m["text"] for m in telegram.messages] == ["first", "second"]      # order kept, nothing lost
    assert waits == [10.0, 30.0, 60.0]                                          # growing pauses
    assert sum("retrying in" in r.message for r in caplog.records) == 3
    assert all(r.levelname == "WARNING" for r in caplog.records)


def test_unreachable_telegram_never_blocks_or_crashes(caplog):
    waits, release = [], threading.Event()

    def pause(seconds):            # the 3rd pause blocks until the test ends
        waits.append(seconds)
        if len(waits) >= 3:
            release.wait(5)

    s = TelegramSender("123:ABC", "-1001", api_url="http://127.0.0.1:9", timeout_s=3, sleep=pause)  # nothing listens
    with caplog.at_level(logging.WARNING, logger="detector.notify"):
        s.send_async("lost in the network")      # returns at once
        for _ in range(250):
            if len(waits) >= 3:
                break
            threading.Event().wait(0.02)
    s.close()
    release.set()
    assert waits[:3] == [10.0, 30.0, 60.0] and s.pending == 1 and s.sent == 0
    assert "could not reach Telegram" in s.last_error
    assert any("could not reach Telegram" in r.message for r in caplog.records)


def test_rate_limit_waits_as_long_as_telegram_asks(telegram):
    telegram.plan, telegram.retry_after = [429], 7
    s, waits = sender(telegram.url)
    s.send_async("busy")
    assert s.flush() and waits == [7.0] and len(telegram.messages) == 1


def test_wrong_chat_or_token_is_not_retried(telegram, caplog):
    telegram.plan = [400]
    s, waits = sender(telegram.url)
    with caplog.at_level(logging.WARNING, logger="detector.notify"):
        s.send_async("to nowhere")
        s.send_async("next one still goes out")
        assert s.flush()
    assert waits == [] and s.dropped == 1
    assert [m["text"] for m in telegram.messages] == ["next one still goes out"]
    assert any("chat not found" in r.message and "not be retried" in r.message for r in caplog.records)
    with pytest.raises(TelegramError) as e:
        telegram.plan = [401]
        s.send("sync")
    assert e.value.retry is False and "Unauthorized" in str(e.value)


def test_old_messages_are_dropped(telegram, monkeypatch):
    s, _ = sender(telegram.url, max_age_s=60)
    s._queue.append((0.0, "from 1970"))      # queued long ago
    s.send_async("fresh")
    assert s.flush()
    assert [m["text"] for m in telegram.messages] == ["fresh"] and s.dropped == 1


# ── POS wording ────────────────────────────────────────────────────────────────────────
def logs_of(lines):
    return records_to_frame(iter_records(lines))


@pytest.fixture(scope="module")
def baseline():
    return fit_baseline(build_windows(logs_of(pos_lines(3600, seed=4)), profile=POS), profile=POS)


def worst_window(kind, column, baseline):
    logs = logs_of(pos_lines(600, seed=12, fault=(300, 420, kind)))
    w = build_windows(logs, profile=POS)
    row = w.iloc[w[column].idxmax()]
    return row, explain(row, baseline, logs, profile=POS)


@pytest.mark.parametrize("kind, column, service, problem", [
    ("WEBHOOK", "system__webhook_retries_exhausted", "pos-worker", "webhook delivery failures"),
    ("SLOWDB", "system__db_query_slow", "pos-api", "slow database"),
    ("PAYMENT", "system__payment_failed", "pos-api", "payment failures"),
    ("LATENCY", "pos-api__latency_p95", "pos-api", "slow responses"),
    ("ERROR", "pos-api__http_5xx", "pos-api", "errors (HTTP 5xx / exceptions)"),
])
def test_pos_explanations_name_the_kind_of_problem(kind, column, service, problem, baseline):
    _, e = worst_window(kind, column, baseline)
    assert e["suspected_service"] == service
    assert problem in e["suspected_problem"]
    assert "like LATENCY" not in e["suspected_problem"] and "DB_TIMEOUT" not in e["suspected_problem"]
    assert e["example_errors"] or kind == "LATENCY"


def test_slow_database_also_reports_the_slow_responses(baseline):
    _, e = worst_window("SLOWDB", "system__db_query_slow", baseline)
    assert e["suspected_problem"].startswith("slow responses")


def test_store_is_named_when_most_failures_come_from_one_store(baseline):
    row, e = worst_window("PAYMENT", "system__payment_failed", baseline)
    assert e["store"]["store"] == "Siem Reap - Pub Street" and e["store"]["store_id"] == "s2"
    assert e["store"]["lines"] / e["store"]["of"] >= 0.6
    # not when failures are spread over the stores or there are no store fields (webhooks)
    _, w = worst_window("WEBHOOK", "system__webhook_retries_exhausted", baseline)
    assert w["store"] is None
    logs = logs_of(pos_lines(60, seed=1))
    assert main_store(logs, 0, 1e12) is None


def test_ecommerce_wording_is_unchanged():
    row = build_windows(logs_of(pos_lines(120, seed=3)), profile=POS).iloc[0]
    e = explain(row, default_baseline(profile="ecommerce"), profile="ecommerce")
    assert "store" not in e


def test_pos_telegram_message(baseline):
    row, e = worst_window("PAYMENT", "system__payment_failed", baseline)
    result = {"window_end_epoch": float(row["window_end"]), "explanation": e,
              "methods": {"rules": {"score": 1.6, "threshold": 1.0, "alert": True, "fired_rules": ["events_payment_failed"]},
                          "pca": {"score": 9.1, "threshold": 1.3, "alert": True}}}
    text = format_detected(result, ["rules", "pca"], 4, "POS", "Asia/Phnom_Penh")
    lines = text.splitlines()
    assert lines[0] == "🚨 Anomaly detected in the POS (incident #4)"
    assert lines[1].endswith("Asia/Phnom_Penh") and "13:0" in lines[1]   # 06:0x UTC = 13:0x in Phnom Penh
    assert "Detected by: Rule-based (default) (1.60), PCA (reconstruction) (9.10)" in text
    assert "Suspected service: pos-api" in text and "Suspected problem: " in text and "payment failures" in text
    assert "Store: Siem Reap - Pub Street (" in text
    assert "Strongest signals:" in text and "failed payments" in text
    assert "Example log lines:" in text and "[pos-api] Payment failed event=payment_failed" in text


# ── live loop with the mock Telegram ────────────────────────────────────────────────────
def test_live_detector_sends_pos_incident_and_keeps_running_when_telegram_fails(tmp_path, telegram):
    logs = write_pos_logs(tmp_path / "train", 3600, seed=4)
    live = tmp_path / "live"
    live.mkdir()
    settings = Settings(profile="pos", log_dir=str(live), model_dir=str(tmp_path / "models"), live=False,
                        labels_path=str(tmp_path / "none.csv"), tail_from_start=True,
                        telegram_token="123:ABC", telegram_chat_id="-1001", telegram_api_url=telegram.url,
                        notify_resolve_after_s=60, notify_timezone="Asia/Phnom_Penh")
    with TestClient(create_app(settings)) as c:
        assert c.post("/train", json={"logs_dir": str(logs)}).status_code == 200
        state = c.app.state.detector
        state.telegram._sleep = lambda s: None          # retry at once in the test
        telegram.plan = [503, 503]                      # Telegram is down when the incident starts

        lines = pos_lines(900, seed=13, fault=(300, 480, "WEBHOOK"))
        f = live / "pos-api.log"
        cut = next(i for i, l in enumerate(lines) if json.loads(l)["@timestamp"] >= "2026-11-12T06:07:30")
        f.write_text("\n".join(lines[:cut]) + "\n")
        body = c.get("/detect").json()
        assert body["status"] == "anomaly"
        assert state.telegram.flush(5)
        assert len(telegram.messages) == 1 and telegram.requests == 3   # 2 failures, then delivered
        text = telegram.messages[0]["text"]
        assert text.startswith("🚨 Anomaly detected in the POS (incident #1)")
        assert "Suspected service: pos-worker" in text and "webhook delivery failures" in text

        f.write_text("\n".join(lines) + "\n")          # the rest: the fault is over
        c.get("/detect")
        assert state.telegram.flush(5)
        assert len(telegram.messages) == 2 and telegram.messages[1]["text"].startswith("✅ Resolved in the POS (incident #1)")
        health = c.get("/health").json()
        assert health["telegram"] == {"pending": 0, "sent": 2, "dropped": 0, "last_error": None}


def test_notify_test_endpoint_with_mock_telegram(tmp_path, telegram):
    settings = Settings(profile="pos", log_dir=str(tmp_path), model_dir=str(tmp_path / "m"), live=False,
                        telegram_token="123:ABC", telegram_chat_id="-1001", telegram_api_url=telegram.url)
    with TestClient(create_app(settings)) as c:
        assert c.post("/notify/test").json() == {"sent": True}
        assert "for the POS" in telegram.messages[0]["text"]
        telegram.plan = [400]
        r = c.post("/notify/test")
        assert r.status_code == 502 and "chat not found" in r.json()["detail"]
