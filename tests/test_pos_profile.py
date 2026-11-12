"""The ``pos`` detector profile: POS services, business events, rules and the API."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app
from detector.explain import default_baseline, explain, fit_baseline
from detector.features import build_windows, feature_columns
from detector.models import IsolationForestDetector, PCADetector, RuleBasedDetector, RuleConfig, symptom_columns
from detector.parsing import iter_records, load_logs, records_to_frame
from detector.profiles import get_profile

from pos_logs import pos_lines, write_pos_logs

POS = get_profile("pos")


def logs_of(lines):
    return records_to_frame(iter_records(lines))


# ── profile ────────────────────────────────────────────────────────────────────────────
def test_profiles_and_default(monkeypatch):
    monkeypatch.delenv("DETECTOR_PROFILE", raising=False)
    assert get_profile().name == "ecommerce"
    assert get_profile().services == ("api-gateway", "order-service", "inventory-service", "payment-service")
    monkeypatch.setenv("DETECTOR_PROFILE", "pos")
    assert get_profile() is POS
    assert POS.services == ("pos-api", "pos-worker") and POS.entry_service == "pos-api"
    with pytest.raises(ValueError, match="unknown DETECTOR_PROFILE"):
        get_profile("shop")


def test_pos_feature_columns_have_no_saga_features():
    cols = feature_columns(profile=POS)
    assert "pos-api__latency_p95" in cols and "pos-worker__kafka_p95" in cols
    for event in ("sale_completed", "payment_failed", "webhook_retries_exhausted", "db_query_slow", "login_failed"):
        assert f"system__{event}" in cols
    assert "system__sale_success_ratio" in cols and "system__unseen_templates" in cols
    assert not [c for c in cols if "saga" in c or "api-gateway" in c]
    # the e-commerce profile is unchanged
    assert "system__saga_success_ratio" in feature_columns(profile="ecommerce")


# ── features ───────────────────────────────────────────────────────────────────────────
def _event(t, event, service="pos-api", level="INFO", **extra):
    return {"@timestamp": pd.Timestamp("2026-11-12T06:00:00Z") + pd.Timedelta(seconds=t), "service": service,
            "level": level, "event": event, "message": f"x event={event}", **extra}


def test_sale_success_ratio_and_event_counts():
    records = [_event(1, "sale_completed"), _event(2, "sale_completed"), _event(3, "sale_completed"),
               _event(4, "sale_failed", level="WARN"), _event(5, "payment_failed", level="WARN"),
               _event(6, "webhook_retries_exhausted", service="pos-worker", level="WARN"),
               _event(45, "http_request", status=200, durationMs=10)]
    for r in records:
        r["@timestamp"] = r["@timestamp"].isoformat()
    w = build_windows(records_to_frame(records), window=30, step=10, profile=POS)
    first = w.iloc[0]
    assert first["system__sale_completed"] == 3 and first["system__payment_failed"] == 1
    assert first["system__webhook_retries_exhausted"] == 1
    assert first["system__sale_success_ratio"] == pytest.approx(3 / 5)
    assert first["pos-worker__warn_count"] == 1
    # a window without sales has ratio 1.0
    assert w.iloc[-1]["system__sale_completed"] == 0 and w.iloc[-1]["system__sale_success_ratio"] == 1.0


def test_windows_from_generated_pos_logs():
    w = build_windows(logs_of(pos_lines(300, seed=2)), profile=POS)
    assert len(w) > 20
    assert (w["pos-api__http_count"] > 0).all()
    assert w["pos-worker__kafka_p95"].max() > 0      # webhook jobs (message_processed durationMs)
    assert w["system__sale_success_ratio"].between(0.5, 1.0).all()


# ── rules ──────────────────────────────────────────────────────────────────────────────
def frame(n=5, **cols):
    df = pd.DataFrame({"window_start": np.arange(n) * 10.0, "window_end": np.arange(n) * 10.0 + 30})
    for k, v in cols.items():
        df[k] = v
    return df


def test_pos_rule_defaults():
    c = RuleConfig.for_profile(POS)
    assert (c.gateway_5xx_ratio, c.gateway_p95_ms, c.errors_per_service, c.min_requests, c.consecutive) == (0.05, 1500.0, 5.0, 10, 3)
    assert dict(c.event_limits) == {"payment_failed": 3.0, "webhook_retries_exhausted": 0.0}
    # e-commerce defaults are unchanged
    assert RuleConfig.for_profile("ecommerce") == RuleConfig()


@pytest.mark.parametrize("cols, rule", [
    ({"pos-api__http_count": 20, "pos-api__http_5xx_ratio": 0.10}, "gateway_5xx_ratio"),
    ({"pos-api__latency_p95": 1600.0}, "gateway_p95_latency"),
    ({"pos-worker__error_count": 6}, "errors_pos-worker"),
    ({"system__payment_failed": 4}, "events_payment_failed"),
    ({"system__webhook_retries_exhausted": 1}, "events_webhook_retries_exhausted"),
])
def test_pos_rules_fire(cols, rule):
    rules = RuleBasedDetector(profile=POS)
    X = frame(**cols)
    assert rules.predict(X).tolist() == [0, 0, 1, 1, 1]      # must hold for 3 windows
    assert rule in rules.fired_rules(X)[-1]


@pytest.mark.parametrize("cols", [
    {"pos-api__http_count": 9, "pos-api__http_5xx_ratio": 0.5},     # too little traffic
    {"pos-api__http_count": 20, "pos-api__http_5xx_ratio": 0.05},   # not above 5 %
    {"pos-api__latency_p95": 1500.0},
    {"pos-api__error_count": 5},
    {"system__payment_failed": 3},
    {"system__webhook_retries_exhausted": 0},
    {"api-gateway__latency_p95": 9000.0},                            # e-commerce columns are ignored
])
def test_pos_rules_stay_quiet(cols):
    assert RuleBasedDetector(profile=POS).predict(frame(**cols)).sum() == 0


def test_rules_need_consecutive_windows():
    X = frame(5, **{"system__webhook_retries_exhausted": [0, 1, 0, 1, 0]})
    assert RuleBasedDetector(profile=POS).predict(X).sum() == 0
    single = RuleBasedDetector(RuleConfig.for_profile(POS).__class__(**{**RuleConfig.for_profile(POS).__dict__, "consecutive": 1}), profile=POS)
    assert single.predict(X).tolist() == [0, 1, 0, 1, 0]


# ── unsupervised models and explanation ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def pos_windows():
    normal = build_windows(logs_of(pos_lines(3600, seed=4)), profile=POS)
    faulty_logs = logs_of(pos_lines(600, seed=5, fault=(300, 420, "LATENCY")))
    faulty = build_windows(faulty_logs, profile=POS)
    return normal, faulty, faulty_logs


def test_symptom_columns_follow_the_profile():
    cols = symptom_columns(feature_columns(profile=POS), profile=POS)
    assert "system__payment_failed" in cols and "system__sale_success_ratio" in cols
    assert "system__sale_completed" not in cols and "system__cash_mismatch" not in cols   # activity, not symptoms
    assert not [c for c in cols if "saga" in c]


def _fault_windows(f):
    t0 = f["window_start"].min()
    return ((f["window_end"] > t0 + 330) & (f["window_start"] < t0 + 400)).to_numpy()


@pytest.mark.parametrize("kind", ["LATENCY", "ERROR", "WEBHOOK"])
def test_pca_with_t2_flags_every_pos_fault_kind(kind, pos_windows):
    normal = pos_windows[0]
    model = PCADetector(profile=POS).fit(normal)
    assert model.t2 is True and not [c for c in model.columns_ if "saga" in c]
    faulty = build_windows(logs_of(pos_lines(600, seed=5, fault=(300, 420, kind))), profile=POS)
    assert model.predict(faulty)[_fault_windows(faulty)].mean() > 0.8


def test_pca_false_alarms_on_other_normal_traffic(pos_windows):
    model = PCADetector(profile=POS).fit(pos_windows[0])
    other_normal = build_windows(logs_of(pos_lines(1800, seed=9)), profile=POS)
    assert model.predict(other_normal).mean() < 0.10     # 97th-percentile threshold, 1 h of training


def test_pca_without_t2_misses_a_latency_only_fault(pos_windows):
    """Why the POS profile turns T² on: the latency direction is inside the PCA subspace."""
    normal, faulty, _ = pos_windows
    spe_only = PCADetector(profile=POS, t2=False).fit(normal)
    assert spe_only.predict(faulty)[_fault_windows(faulty)].mean() < 0.5
    assert get_profile("ecommerce").pca_t2 is False     # thesis behaviour unchanged


def test_isolation_forest_runs_on_pos_windows(pos_windows):
    normal, faulty, _ = pos_windows
    model = IsolationForestDetector(profile=POS).fit(normal)
    assert not [c for c in model.columns_ if "saga" in c]
    assert model.score(faulty).shape == (len(faulty),)


def test_explanation_names_pos_api_and_slow_responses(pos_windows):
    normal, faulty, faulty_logs = pos_windows
    baseline = fit_baseline(normal, profile=POS)
    assert "pos-api__latency_p95" in baseline and "system__sale_success_ratio" in baseline
    row = faulty.iloc[(faulty["pos-api__latency_p95"]).idxmax()]
    e = explain(row, baseline, faulty_logs, profile=POS)
    assert e["suspected_service"] == "pos-api"
    assert "slow responses" in e["suspected_problem"]
    assert any("pos-api: p95 latency" in s["text"] for s in e["top_signals"])


def test_explanation_of_webhook_failures():
    logs = logs_of(pos_lines(400, seed=6, fault=(200, 300, "WEBHOOK")))
    w = build_windows(logs, profile=POS)
    row = w.iloc[w["system__webhook_retries_exhausted"].idxmax()]
    e = explain(row, default_baseline(profile=POS), logs, profile=POS)
    assert e["suspected_service"] == "pos-worker"
    assert "webhook deliveries were given up after all retries" in e["notes"]
    assert e["example_errors"] and e["example_errors"][0].startswith("[pos-worker]")


# ── API ────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def pos_client(tmp_path):
    logs = write_pos_logs(tmp_path / "logs", 1800, seed=8)
    settings = Settings(profile="pos", log_dir=str(logs), labels_path=str(tmp_path / "none.csv"),
                        results_dir=str(tmp_path / "results"), model_dir=str(tmp_path / "models"),
                        live=False, tail_from_start=True, buffer_minutes=10)
    with TestClient(create_app(settings)) as c:
        yield c


def test_api_with_pos_profile(pos_client, tmp_path):
    assert pos_client.get("/health").json()["profile"] == "pos"
    body = pos_client.get("/detect").json()
    assert body["status"] in ("normal", "anomaly")
    assert {"pos-api__latency_p95", "pos-worker__error_count", "system__sale_success_ratio",
            "system__payment_failed"} <= set(body["features"])
    assert not [k for k in body["features"] if "saga" in k or "api-gateway" in k]

    trained = pos_client.post("/train", json={}).json()
    assert trained["faults"] == 0 and trained["random_forest"] is False     # no labels: unsupervised only
    assert pos_client.get("/detect").json()["methods"].keys() >= {"rules", "isolation_forest"}


def test_model_of_another_profile_is_not_loaded(tmp_path, pos_client):
    pos_client.post("/train", json={})
    settings = Settings(profile="ecommerce", log_dir=str(tmp_path / "logs"), model_dir=str(tmp_path / "models"),
                        results_dir=str(tmp_path / "results"), live=False)
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["models"]["isolation_forest"] is False


def test_load_logs_reads_the_pos_log_file(tmp_path):
    folder = write_pos_logs(tmp_path / "logs", 120)
    logs = load_logs(folder)
    assert set(logs["service"]) == {"pos-api", "pos-worker"}
