"""Live detection with the POS profile: training without labels, PCA, reload, API key, scores."""
import json

import pytest
from fastapi.testclient import TestClient

from detector import cli
from detector.api import Settings, create_app

from pos_logs import write_pos_logs


def make_settings(tmp_path, **overrides):
    values = dict(profile="pos", log_dir=str(tmp_path / "logs"), labels_path=str(tmp_path / "no-labels.csv"),
                  results_dir=str(tmp_path / "results"), model_dir=str(tmp_path / "models"),
                  live=False, tail_from_start=True, buffer_minutes=10)
    values.update(overrides)
    return Settings(**values)


@pytest.fixture()
def pos_logs(tmp_path):
    # 2 hours; webhook deliveries fail between minute 100 and 103
    return write_pos_logs(tmp_path / "logs", 7200, seed=11, fault=(6000, 6180, "WEBHOOK"))


def test_train_without_labels_uses_unsupervised_models_and_the_rules(tmp_path, pos_logs):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        result = c.post("/train", json={}).json()
        assert result["pca"] is True and result["random_forest"] is False and result["faults"] == 0
        assert result["excluded_windows"] > 0          # the webhook incident is not learned as normal
        assert "warning" in result and "24" in result["warning"]   # 2 h of logs < 1 day
        health = c.get("/health").json()
        assert health["models"] == {"rules": True, "isolation_forest": True, "pca": True, "random_forest": False,
                                    "trained_at": health["models"]["trained_at"]}
        assert health["notify_methods"] == ["rules", "pca"]
        body = c.get("/detect").json()
        assert set(body["methods"]) == {"rules", "isolation_forest", "pca"}
        assert body["methods"]["pca"]["threshold"] > 0


def test_train_on_the_last_hours(tmp_path, pos_logs):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        result = c.post("/train", json={"hours": 1}).json()
        assert 0.9 <= result["hours"] <= 1.01
        assert result["excluded_windows"] > 0          # the incident is in the last hour


def test_notify_methods_can_be_overridden(tmp_path, pos_logs):
    with TestClient(create_app(make_settings(tmp_path, notify_methods=("rules",)))) as c:
        assert c.get("/health").json()["notify_methods"] == ["rules"]


def test_cli_train_and_the_running_detector_reloads(tmp_path, pos_logs, capsys):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/health").json()["models"]["pca"] is False
        code = cli.main(["--profile", "pos", "train", "--logs", str(pos_logs), "--models", str(tmp_path / "models"),
                         "--hours", "1.5"])
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["pca"] is True
        c.get("/detect")                              # a poll picks up the new model file
        assert c.get("/health").json()["models"]["pca"] is True


def test_cli_train_reports_missing_logs(tmp_path, capsys):
    assert cli.main(["--profile", "pos", "train", "--logs", str(tmp_path / "empty"),
                     "--models", str(tmp_path / "m")]) == 2
    assert "no log records" in capsys.readouterr().err


def test_api_key_protects_everything_but_health(tmp_path, pos_logs):
    with TestClient(create_app(make_settings(tmp_path, api_key="s3cret-key"))) as c:
        assert c.get("/health").status_code == 200
        for path in ("/alerts", "/scores", "/detect"):
            assert c.get(path).status_code == 401
            assert c.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/notify/test").status_code == 401
        assert c.get("/alerts", headers={"X-API-Key": "s3cret-key"}).status_code == 200


def test_scores_after_polls(tmp_path, pos_logs):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        c.post("/train", json={})
        c.get("/detect")
        body = c.get("/scores", params={"minutes": 30}).json()
        assert {"rules", "isolation_forest", "pca"} <= set(body["methods"])
        point = body["points"][-1]
        assert point["time"].endswith("+00:00")
        assert set(point["methods"]["pca"]) == {"score", "threshold", "alert"}
