import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app


@pytest.fixture()
def client(synthetic_run, tmp_path):
    settings = Settings(log_dir=str(synthetic_run / "logs"), labels_path=str(synthetic_run / "labels.csv"),
                        results_dir=str(tmp_path / "results"), model_dir=str(tmp_path / "models"),
                        live=False, tail_from_start=True, buffer_minutes=10)
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "UP"
    assert body["models"] == {"rules": True, "isolation_forest": False, "random_forest": False, "trained_at": None}


def test_detect_with_rules_only_before_training(client):
    body = client.get("/detect").json()
    assert body["status"] in ("normal", "anomaly")
    assert set(body["methods"]) == {"rules"}
    assert "api-gateway__latency_p95" in body["features"]


def test_train_then_detect_uses_all_models(client, tmp_path):
    trained = client.post("/train", json={}).json()
    assert trained["random_forest"] is True and trained["anomalous_windows"] > 0
    assert (tmp_path / "models" / "detector.joblib").exists()
    assert client.get("/health").json()["models"]["isolation_forest"] is True

    body = client.get("/detect").json()
    assert set(body["methods"]) == {"rules", "isolation_forest", "random_forest"}
    for info in body["methods"].values():
        assert isinstance(info["alert"], bool)

    alerts = client.get("/alerts", params={"limit": 5}).json()
    assert alerts["count"] == len(alerts["alerts"]) <= 5


def test_evaluate_endpoint_writes_report(client, tmp_path):
    response = client.post("/evaluate", json={"make_figures": False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["metrics"]) == {"rules", "isolation_forest", "random_forest"}
    assert "report.md" in body["outputs"]
    assert (tmp_path / "results" / "results.json").exists()


def test_evaluate_rejects_missing_labels(client, tmp_path):
    response = client.post("/evaluate", json={"labels_path": str(tmp_path / "nope.csv")})
    assert response.status_code == 400


def test_train_rejects_empty_log_dir(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    response = client.post("/train", json={"logs_dir": str(empty)})
    assert response.status_code == 400


def test_cors_allows_the_flutter_web_app(client):
    pre = client.options("/api/services", headers={"Origin": "http://localhost:5000",
                                                   "Access-Control-Request-Method": "GET"})
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] in ("*", "http://localhost:5000")
    got = client.get("/health", headers={"Origin": "http://localhost:5000"})
    assert got.headers["access-control-allow-origin"] in ("*", "http://localhost:5000")


def test_cors_origins_can_be_restricted(synthetic_run, tmp_path):
    settings = Settings(log_dir=str(synthetic_run / "logs"), results_dir=str(tmp_path / "r"),
                        model_dir=str(tmp_path / "m"), live=False, cors_allow_origins=("http://phone.local:5000",))
    with TestClient(create_app(settings)) as c:
        ok = c.get("/health", headers={"Origin": "http://phone.local:5000"})
        assert ok.headers.get("access-control-allow-origin") == "http://phone.local:5000"
        other = c.get("/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in other.headers
