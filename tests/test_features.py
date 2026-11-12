import numpy as np
import pandas as pd
import pytest

from detector.features import (
    add_unseen_templates, build_windows, feature_columns, feature_name, label_windows, system_name,
    window_starts,
)
from detector.parsing import records_to_frame

T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def rec(sec, service="api-gateway", level="INFO", message="HTTP request", **fields):
    return {"@timestamp": (T0 + pd.Timedelta(seconds=sec)).isoformat(), "service": service, "level": level,
            "message": message, **fields}


def labels(*faults):
    return pd.DataFrame([{"fault_id": f"F{i}", "service": "order-service", "type": "ERROR",
                          "start": T0 + pd.Timedelta(seconds=s), "end": T0 + pd.Timedelta(seconds=e), "params": ""}
                         for i, (s, e) in enumerate(faults, 1)])


def test_window_starts_are_aligned_and_complete():
    starts = window_starts(T0.timestamp() + 3, T0.timestamp() + 95, 30, 10)
    assert starts[0] == T0.timestamp()
    assert starts[-1] + 30 <= T0.timestamp() + 95
    assert np.allclose(np.diff(starts), 10)


def test_build_windows_computes_service_and_system_features():
    records = []
    for i in range(10):     # 10 gateway requests in [0, 10): 2 are 5xx, durations 100..1000 ms
        status = 500 if i < 2 else 200
        records.append(rec(i + 0.5, level="ERROR" if status == 500 else "INFO", event="http_request",
                           status=status, durationMs=(i + 1) * 100))
    records.append(rec(1, service="order-service", level="ERROR", message="Unhandled exception",
                       event="unhandled_exception", stack_trace="x"))
    records.append(rec(2, service="order-service", level="WARN", message="Saga timed out", event="saga_timeout"))
    for s in range(4):
        records.append(rec(3 + s, service="order-service", message="Saga started", event="saga_started"))
    records.append(rec(8, service="order-service", message="Saga completed", event="saga_completed"))
    records.append(rec(9, service="inventory-service", message="Message processed", event="message_processed",
                       durationMs=40))
    records.append(rec(75, service="payment-service", message="tail"))

    logs = records_to_frame(records)
    w = build_windows(logs, window=30, step=10)
    first = w.iloc[0]
    assert list(w.columns[2:]) == feature_columns()
    gw = lambda f: first[feature_name("api-gateway", f)]
    assert gw("http_count") == 10 and gw("http_5xx") == 2
    assert gw("http_5xx_ratio") == pytest.approx(0.2)
    assert gw("latency_max") == 1000 and gw("latency_mean") == pytest.approx(550)
    assert gw("latency_p95") == pytest.approx(np.percentile(np.arange(1, 11) * 100, 95))
    assert first[feature_name("order-service", "exception_count")] == 1
    assert first[feature_name("order-service", "error_count")] == 1
    assert first[feature_name("order-service", "warn_count")] == 1
    assert first[feature_name("inventory-service", "kafka_p95")] == pytest.approx(40)
    assert first[system_name("saga_started")] == 4
    assert first[system_name("saga_timeout")] == 1
    assert first[system_name("saga_success_ratio")] == pytest.approx(0.25)
    # a window with no sagas is neutral
    assert w.iloc[-1][system_name("saga_success_ratio")] == 1.0


def test_label_windows_uses_overlap_with_grace():
    logs = records_to_frame([rec(s) for s in range(0, 200, 2)])
    w = build_windows(logs, window=30, step=10)
    lw = label_windows(w, labels((100, 120)), grace=15)
    ws, we = lw["window_start"] - T0.timestamp(), lw["window_end"] - T0.timestamp()
    expected = ((ws < 135) & (we > 100)).astype(int)
    assert list(lw["label"]) == list(expected)
    assert set(lw.loc[lw["label"] == 1, "fault_id"]) == {"F1"}
    # window [70, 100) ends exactly at the fault start: not anomalous
    assert lw.loc[ws == 70, "label"].item() == 0
    # window [130, 160) overlaps the grace period: anomalous
    assert lw.loc[ws == 130, "label"].item() == 1


def test_unseen_templates_counts_lines_not_in_training_templates():
    logs = records_to_frame([rec(1, message="normal id=1"), rec(2, message="normal id=2"),
                             rec(40, level="ERROR", message="Connection reset by peer"),
                             rec(42, message="a new kind of normal request"),     # new INFO: not counted
                             rec(41, message="normal id=3"), rec(70, message="end")])
    w = build_windows(logs, window=30, step=10)
    known = set(logs.loc[logs["t"] < T0.timestamp() + 30, "template"])
    w = add_unseen_templates(w, logs, known)
    col = system_name("unseen_templates")
    assert w.loc[w["window_start"] == T0.timestamp(), col].item() == 0
    assert w.loc[w["window_start"] == T0.timestamp() + 20, col].item() == 1
