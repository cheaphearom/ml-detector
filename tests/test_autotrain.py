"""Automatic training: when it runs, what it leaves out, when it refuses, and the live loop running it."""
import datetime as dt
import json
import time

import pytest
from fastapi.testclient import TestClient

import detector.api as api_module
from detector.api import Settings, create_app
from detector.autotrain import AutoTrainer, IncidentLog

from pos_logs import START, write_pos_logs

T0 = START.timestamp()


def local(y, mo, d, h, mi, tz="Asia/Phnom_Penh"):
    from zoneinfo import ZoneInfo
    return dt.datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp()


# ── when ────────────────────────────────────────────────────────────────────────────────
def test_first_model_as_soon_as_possible_at_the_set_time():
    a = AutoTrainer(at="04:30", timezone="Asia/Phnom_Penh")
    assert not a.due(local(2026, 11, 20, 4, 29), None, False)          # before 04:30
    assert a.due(local(2026, 11, 20, 4, 31), None, False)              # no model yet: train
    assert not a.due(local(2026, 11, 20, 9, 0), None, False)           # once a day at most
    assert a.next_try(local(2026, 11, 20, 9, 0), None).startswith("2026-11-21T04:30")


def test_then_every_week():
    a = AutoTrainer(every_days=7, at="04:30", timezone="Asia/Phnom_Penh")
    trained = dt.datetime.fromtimestamp(local(2026, 11, 20, 4, 35), tz=dt.timezone.utc).isoformat()
    for day in range(21, 27):
        assert not a.due(local(2026, 11, day, 4, 31), trained, False), day
    assert a.due(local(2026, 11, 27, 4, 31), trained, False)           # 7 days later (a few minutes early is fine)
    assert a.next_try(local(2026, 11, 22, 12, 0), trained).startswith("2026-11-27T04:30")


def test_postponed_while_an_incident_is_open():
    a = AutoTrainer(at="04:30", timezone="UTC")
    assert not a.due(dt.datetime(2026, 11, 20, 5, tzinfo=dt.timezone.utc).timestamp(), None, True)
    assert a.last["ok"] is False and "incident is open" in a.last["message"]
    assert a.due(dt.datetime(2026, 11, 21, 5, tzinfo=dt.timezone.utc).timestamp(), None, False)   # next day


# ── the record of alerted periods ─────────────────────────────────────────────────────────
def test_incident_log_merges_keeps_30_days_and_survives_a_restart(tmp_path):
    path = str(tmp_path / "incidents.json")
    log = IncidentLog(path, keep_days=30, margin_s=60)
    log.add(1000, 1030)
    log.add(1040, 1070)              # overlaps (with the margin): one period
    log.add(5000, 5030)
    assert log.periods == [[940, 1130], [4940, 5090]]
    assert IncidentLog(path).periods == log.periods
    log.add(5000 + 31 * 86400, 5030 + 31 * 86400)
    assert len(log.periods) == 1     # older than 30 days: dropped


# ── training with the safety options ─────────────────────────────────────────────────────
def settings(tmp_path, **kw):
    values = dict(profile="pos", log_dir=str(tmp_path / "logs"), labels_path=str(tmp_path / "none.csv"),
                  results_dir=str(tmp_path / "results"), model_dir=str(tmp_path / "models"), live=False,
                  tail_from_start=True)
    values.update(kw)
    return Settings(**values)


def test_training_leaves_out_alerted_periods_and_refuses_unusual_or_short_periods(tmp_path):
    write_pos_logs(tmp_path / "logs", 3600, seed=3)
    with TestClient(create_app(settings(tmp_path))) as c:
        state = c.app.state.detector
        plain = state.train(None, None)
        alerted = state.train(None, None, exclude_periods=[[T0 + 600, T0 + 900]])
        assert alerted["excluded_windows"] >= plain["excluded_windows"] + 25      # 5 minutes of 10-s windows
        with pytest.raises(ValueError, match="looked unusual"):
            state.train(None, None, exclude_periods=[[T0, T0 + 1800]], max_unusual=0.2)
        with pytest.raises(ValueError, match="hours of logs so far"):
            state.train(None, None, min_hours=2)


# ── the live loop ───────────────────────────────────────────────────────────────────────
def test_live_loop_trains_by_itself_in_the_background_and_says_so(tmp_path, monkeypatch):
    write_pos_logs(tmp_path / "logs", 3600, seed=5)
    sent = []
    clock = [dt.datetime(2026, 11, 20, 5, 0, tzinfo=dt.timezone.utc).timestamp()]
    monkeypatch.setattr(api_module.time, "time", lambda: clock[0])
    monkeypatch.setattr("detector.autotrain.dt.datetime", dt.datetime)
    s = settings(tmp_path, auto_train=True, auto_train_at="04:30", auto_train_hours=1, min_train_hours=0.5,
                 notify_timezone="UTC")
    with TestClient(create_app(s, send=sent.append)) as c:
        state = c.app.state.detector
        assert state.trained_at is None
        c.get("/detect")
        for _ in range(300):                       # the training runs in a background thread
            if not state.autotrain.running and state.autotrain.last:
                break
            time.sleep(0.1)
        assert state.autotrain.last["ok"] is True, state.autotrain.last
        assert state.trained_at is not None and state.model_path().exists()
        assert any("trained for the first time" in m for m in sent)
        auto = c.get("/health").json()["auto_training"]
        # 7 days after the training (its time is the real clock), at 04:30
        week = (dt.datetime.fromisoformat(state.trained_at) + dt.timedelta(days=7)).astimezone(dt.timezone.utc).date()
        assert auto["last"]["ok"] is True and auto["next_try"].startswith(f"{week.isoformat()}T04:30")
        c.get("/detect")                            # same day: no second training
        assert sum("trained" in m for m in sent) == 1


def test_off_by_default(tmp_path):
    (tmp_path / "logs").mkdir()
    with TestClient(create_app(settings(tmp_path))) as c:
        assert c.get("/health").json()["auto_training"] is None
