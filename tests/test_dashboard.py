import json

import pytest
from fastapi.testclient import TestClient

from detector.api import Settings, create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    results = tmp_path / "results"
    run = results / "seed1"
    (run / "detector").mkdir(parents=True)
    (run / "labels.csv").write_text("fault_id,service,type,start,end,params\n"
                                    "F001,payment-service,ERROR,2026-01-01T00:10:00Z,2026-01-01T00:12:00Z,{}\n")
    (run / "experiment.json").write_text(json.dumps({"seed": 1, "completed": True}))
    (run / "detector" / "results.json").write_text(json.dumps({
        "metrics": {"rules": {"f1": 0.5}}, "detection_delay": {"rules": {"detected": 1, "faults": 1}},
        "comparison_vs_rules": {}}))
    (run / "detector" / "report.md").write_text("# report")
    monkeypatch.setenv("RESULTS_ROOT", str(results))
    for name in ("GATEWAY_URL", "ORDER_SERVICE_URL", "INVENTORY_SERVICE_URL", "PAYMENT_SERVICE_URL"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")          # nothing listens there
    settings = Settings(log_dir=str(tmp_path / "logs"), model_dir=str(tmp_path / "models"),
                        results_dir=str(results / "detector"), live=False)
    with TestClient(create_app(settings)) as c:
        c.results = results
        yield c


def test_dashboard_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Anomaly Monitor" in response.text


def test_services_reports_unreachable_services(client):
    body = client.get("/api/services").json()
    assert {s["name"]: s["health"] for s in body["services"]} == {
        "api-gateway": "UNREACHABLE", "order-service": "UNREACHABLE",
        "inventory-service": "UNREACHABLE", "payment-service": "UNREACHABLE"}
    assert body["experiment_running"] is None


def test_experiments_list_and_result_files(client):
    exp = client.get("/api/experiments").json()["experiments"]
    assert len(exp) == 1 and exp[0]["name"] == "seed1"
    assert exp[0]["status"] == "completed" and exp[0]["faults"] == 1
    assert exp[0]["evaluation"]["metrics"]["rules"]["f1"] == 0.5
    assert client.get("/files/seed1/detector/report.md").text == "# report"


def test_faults_are_refused_while_an_experiment_runs(client):
    (client.results / ".experiment-running").write_text(json.dumps({"out": "/results/seed2"}))
    response = client.post("/api/chaos/inventory-service", json={"type": "ERROR"})
    assert response.status_code == 409
    assert "experiment is running" in response.json()["detail"]
    assert client.delete("/api/chaos/inventory-service").status_code == 409
    assert client.get("/api/services").json()["experiment_running"]["out"] == "/results/seed2"


def test_fault_requests_are_validated(client):
    assert client.post("/api/chaos/api-gateway", json={"type": "ERROR"}).status_code == 404
    assert client.post("/api/chaos/inventory-service", json={"type": "BOOM"}).status_code == 422
    assert client.post("/api/chaos/inventory-service", json={"type": "ERROR"}).status_code == 502   # unreachable


def test_history_starts_empty(client):
    assert client.get("/api/history").json() == {"points": []}
