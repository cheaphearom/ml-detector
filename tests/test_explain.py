import pandas as pd

from detector.explain import default_baseline, explain, fit_baseline
from detector.features import (
    add_unseen_templates, build_windows, feature_columns, feature_name, in_fault_mask, known_templates,
    label_windows, system_name,
)
from detector.parsing import load_labels, load_logs


def normal_window(**overrides):
    row = {c: 0.0 for c in feature_columns()}
    for s in ("api-gateway", "order-service", "inventory-service", "payment-service"):
        row[feature_name(s, "latency_p95")] = 20.0
        row[feature_name(s, "latency_mean")] = 10.0
    row[system_name("saga_success_ratio")] = 0.9
    row.update({"window_start": 0.0, "window_end": 30.0})
    row.update(overrides)
    return pd.Series(row)


def test_normal_window_names_no_service():
    ex = explain(normal_window())
    assert ex["suspected_service"] is None
    assert ex["top_signals"] == []


def test_errors_in_one_service():
    ex = explain(normal_window(**{feature_name("payment-service", "error_count"): 40,
                                  feature_name("payment-service", "exception_count"): 35}))
    assert ex["suspected_service"] == "payment-service"
    assert "ERROR" in ex["suspected_problem"]
    assert ex["top_signals"][0]["feature"].startswith("payment-service__")


def test_latency_only():
    ex = explain(normal_window(**{feature_name("inventory-service", "kafka_p95"): 3000,
                                  feature_name("api-gateway", "latency_p95"): 2500}))
    assert ex["suspected_service"] == "inventory-service"
    assert "LATENCY" in ex["suspected_problem"]


def test_latency_and_errors_point_to_database():
    ex = explain(normal_window(**{feature_name("inventory-service", "latency_p95"): 4000,
                                  feature_name("inventory-service", "error_count"): 30,
                                  system_name("saga_timeout"): 5}))
    assert ex["suspected_service"] == "inventory-service"
    assert "DB_TIMEOUT" in ex["suspected_problem"]
    assert any("saga timeouts" in n for n in ex["notes"])


def test_example_errors_come_from_the_window_and_suspected_service():
    logs = pd.DataFrame({
        "t": [5.0, 6.0, 7.0, 50.0],
        "service": ["payment-service", "order-service", "payment-service", "payment-service"],
        "level": ["ERROR", "ERROR", "ERROR", "ERROR"],
        "event": ["message_failed", "message_failed", "message_failed", "message_failed"],
        "message": ["charge failed A", "order failed", "charge failed B", "outside window"],
        "template": ["a", "b", "c", "d"],
    })
    ex = explain(normal_window(**{feature_name("payment-service", "error_count"): 40}), logs=logs)
    assert ex["example_errors"] == ["[payment-service] charge failed A", "[payment-service] charge failed B"]


def test_names_the_injected_service_and_type_on_synthetic_faults(synthetic_run):
    logs = load_logs(synthetic_run / "logs")
    labels = load_labels(synthetic_run / "labels.csv")
    w = label_windows(build_windows(logs), labels, 15)
    w = add_unseen_templates(w, logs, known_templates(logs, ~in_fault_mask(logs["t"].to_numpy(), labels, 15)))
    baseline = fit_baseline(w[w["label"] == 0])
    assert baseline and baseline != default_baseline()

    service_ok = type_ok = 0
    for f in labels.itertuples(index=False):
        s, e = f.start.timestamp(), f.end.timestamp()
        mid = w[(w["window_start"] >= s + 0.5 * (e - s) - 30) & (w["window_end"] <= e)].iloc[-1]
        ex = explain(mid, baseline, logs)
        service_ok += ex["suspected_service"] == f.service
        type_ok += f.type in ex["suspected_problem"]
    assert service_ok >= 0.8 * len(labels)
    assert type_ok >= 0.8 * len(labels)
