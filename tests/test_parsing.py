import gzip
import json

import pandas as pd

from detector.parsing import (
    LogTailer, load_experiment_meta, load_labels, load_logs, make_template, records_to_frame, iter_records,
)


def line(ts, message, level="INFO", service="order-service", **fields):
    return json.dumps({"@timestamp": ts, "message": message, "level": level, "service": service,
                       "logger_name": "x", **fields})


def test_template_masks_variable_parts():
    a = make_template("Saga started event=saga_started sagaId=3f2a9c1e-4b5d-4e6f-8a7b-9c0d1e2f3a4b "
                      "orderId=11111111-2222-4333-8444-555555555555 productId=P-103 quantity=2")
    b = make_template("Saga started event=saga_started sagaId=aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee "
                      "orderId=99999999-8888-4777-8666-555555555555 productId=P-100 quantity=3")
    assert a == b
    assert "<UUID>" in a and "<ID>" in a and "<NUM>" in a


def test_template_of_hikari_timeout_is_stable():
    m1 = ("JDBC query timeout after 2817ms during reserveStock; HikariPool-1 - Connection is not available, "
          "request timed out (Total=10, Active=10, Idle=0, Waiting=17)")
    m2 = m1.replace("2817", "4410").replace("Waiting=17", "Waiting=3")
    assert make_template(m1) == make_template(m2)
    assert "Waiting=<NUM>" in make_template(m1)


def test_template_handles_empty():
    assert make_template("") == ""
    assert make_template(None) == ""


def test_records_to_frame_parses_fields_and_skips_garbage():
    lines = [
        line("2026-01-01T00:00:01.500Z", "HTTP request", event="http_request", status=503, durationMs=120),
        "not json",
        "{broken",
        line("2026-01-01T00:00:00.000Z", "boom", level="ERROR", stack_trace="java.lang.X: y"),
    ]
    df = records_to_frame(iter_records(lines))
    assert len(df) == 2
    assert list(df["t"]) == [pd.Timestamp("2026-01-01", tz="UTC").timestamp(),
                             pd.Timestamp("2026-01-01T00:00:01.5", tz="UTC").timestamp()]
    assert df.iloc[0]["has_stack"] and not df.iloc[1]["has_stack"]
    assert df.iloc[1]["status"] == 503 and df.iloc[1]["durationMs"] == 120


def test_load_logs_reads_plain_and_gzipped_files(tmp_path):
    (tmp_path / "order-service.log").write_text(line("2026-01-01T00:00:02Z", "b") + "\n")
    with gzip.open(tmp_path / "order-service.2026-01-01.0.log.gz", "wt") as fh:
        fh.write(line("2026-01-01T00:00:01Z", "a") + "\n")
    (tmp_path / "ignored.txt").write_text(line("2026-01-01T00:00:03Z", "c") + "\n")
    df = load_logs(tmp_path)
    assert list(df["message"]) == ["a", "b"]


def test_load_labels_and_meta(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("fault_id,service,type,start,end,params\n"
                    'F001,payment-service,ERROR,2026-01-01T00:10:00.000Z,2026-01-01T00:12:00.000Z,"{""errorRate"":0.3}"\n')
    (tmp_path / "experiment.json").write_text(json.dumps({"load_start": "2026-01-01T00:00:00Z"}))
    labels = load_labels(path)
    assert labels.iloc[0]["fault_id"] == "F001"
    assert labels.iloc[0]["end"] - labels.iloc[0]["start"] == pd.Timedelta(minutes=2)
    assert load_experiment_meta(path)["load_start"] == "2026-01-01T00:00:00Z"
    assert load_labels(tmp_path / "missing.csv").empty


def test_log_tailer_reads_incrementally_and_handles_rotation(tmp_path):
    f = tmp_path / "payment-service.log"
    f.write_text(line("2026-01-01T00:00:01Z", "one") + "\n")
    tailer = LogTailer(tmp_path)
    assert list(tailer.read_new()["message"]) == ["one"]
    assert tailer.read_new().empty

    second = line("2026-01-01T00:00:02Z", "two")
    with f.open("a") as fh:
        fh.write(second[:20])           # incomplete line is held back
    assert tailer.read_new().empty
    with f.open("a") as fh:
        fh.write(second[20:] + "\n")
    assert list(tailer.read_new()["message"]) == ["two"]

    f.write_text(line("2026-01-01T00:00:03Z", "after rotation") + "\n")   # file shrank
    assert list(tailer.read_new()["message"]) == ["after rotation"]


def test_log_tailer_from_end_skips_history(tmp_path):
    f = tmp_path / "order-service.log"
    f.write_text(line("2026-01-01T00:00:01Z", "old") + "\n")
    tailer = LogTailer(tmp_path, from_end=True)
    assert tailer.read_new().empty
    with f.open("a") as fh:
        fh.write(line("2026-01-01T00:00:02Z", "new") + "\n")
    assert list(tailer.read_new()["message"]) == ["new"]
