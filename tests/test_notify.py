import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app
from detector.notify import IncidentNotifier, TelegramSender, format_detected
from detector.parsing import load_labels


def result(t, rules=False, rf=False, explanation=None):
    return {
        "window_end_epoch": t,
        "methods": {
            "rules": {"score": 2.0 if rules else 0.1, "threshold": 1.0, "alert": rules,
                      "fired_rules": ["errors_payment-service"] if rules else []},
            "random_forest": {"score": 0.9 if rf else 0.1, "threshold": 0.5, "alert": rf},
        },
        "explanation": explanation,
    }


def test_one_message_per_incident_and_one_when_resolved():
    sent = []
    n = IncidentNotifier(sent.append, ["rules", "random_forest"], resolve_after_s=60)
    assert n.update(result(0)) is None
    assert n.update(result(10, rf=True)) == "opened"
    for t in range(20, 100, 10):                  # still failing: no new messages
        assert n.update(result(t, rules=True, rf=True)) is None
    assert n.update(result(130)) is None          # quiet for 40 s: not yet resolved
    assert n.update(result(150)) == "resolved"    # quiet for 60 s
    assert len(sent) == 2
    assert sent[0].startswith("🚨 Anomaly detected (incident #1)")
    assert sent[1].startswith("✅ Resolved (incident #1)")
    assert "Random Forest" in sent[1] and "Rule-based" in sent[1]
    assert n.update(result(160, rules=True)) == "opened"
    assert "incident #2" in sent[2]


def test_only_selected_methods_trigger_notifications():
    sent = []
    n = IncidentNotifier(sent.append, ["random_forest"])
    assert n.update(result(10, rules=True)) is None
    assert sent == []


def test_message_contains_the_explanation():
    text = format_detected(result(10, rules=True, rf=True, explanation={
        "suspected_service": "payment-service",
        "suspected_problem": "errors / exceptions (like ERROR)",
        "top_signals": [{"text": "payment-service: ERROR log lines = 40 (normal ≈ 0)"}],
        "notes": ["orders are getting stuck (saga timeouts)"],
        "example_errors": ["[payment-service] Message processing failed ... exception=NullPointerException"],
    }), ["rules", "random_forest"], 1)
    for part in ("Suspected service: payment-service", "like ERROR", "ERROR log lines = 40",
                 "NullPointerException", "Rules fired: errors_payment-service", "Random Forest (0.90)"):
        assert part in text


def test_sender_requires_token_and_chat():
    with pytest.raises(ValueError):
        TelegramSender("", "123")


def test_notify_test_endpoint_without_telegram(tmp_path):
    settings = Settings(log_dir=str(tmp_path), model_dir=str(tmp_path / "m"), live=False,
                        telegram_token="", telegram_chat_id="")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["telegram_notifications"] is False
        assert client.post("/notify/test").status_code == 400


def _copy_logs_until(src, dst, cutoff, start=None):
    """Appends the lines of every log file with start < timestamp <= cutoff."""
    dst.mkdir(exist_ok=True)
    for f in src.glob("*.log"):
        with f.open() as fin, (dst / f.name).open("a") as fout:
            for line in fin:
                ts = pd.Timestamp(json.loads(line)["@timestamp"]).timestamp()
                if (start is None or ts > start) and ts <= cutoff:
                    fout.write(line)


def test_live_loop_sends_detected_and_resolved_messages(synthetic_run, tmp_path):
    labels = load_labels(synthetic_run / "labels.csv")
    fault = labels[labels["type"].isin(["ERROR", "DB_TIMEOUT"])].iloc[-1]
    s, e = fault["start"].timestamp(), fault["end"].timestamp()
    live_dir = tmp_path / "live"
    sent = []
    settings = Settings(log_dir=str(live_dir), model_dir=str(tmp_path / "models"), live=False,
                        tail_from_start=True, notify_resolve_after_s=60)
    with TestClient(create_app(settings, send=sent.append)) as client:
        trained = client.post("/train", json={"logs_dir": str(synthetic_run / "logs"),
                                              "labels_path": str(synthetic_run / "labels.csv")})
        assert trained.status_code == 200, trained.text

        cutoff = s + 0.7 * (e - s)                    # in the middle of the fault
        _copy_logs_until(synthetic_run / "logs", live_dir, cutoff)
        body = client.get("/detect").json()
        assert body["status"] == "anomaly"
        assert body["explanation"]["suspected_service"] == fault["service"]
        assert len(sent) == 1 and "Anomaly detected" in sent[0]
        assert f"Suspected service: {fault['service']}" in sent[0]

        _copy_logs_until(synthetic_run / "logs", live_dir, e + 180, start=cutoff)   # well after the fault
        client.get("/detect")
        assert len(sent) == 2 and sent[1].startswith("✅ Resolved")
