"""Monitoring dashboard: the web page at ``/`` and the endpoints it uses.

  GET    /                       the dashboard page
  GET    /api/history            anomaly score per method for the recent windows
  GET    /api/services           health of every service and the fault active on it
  POST   /api/chaos/{service}    inject a fault (refused while an experiment is running)
  DELETE /api/chaos/{service}    clear a fault (refused while an experiment is running)
  GET    /api/experiments        experiment runs in the results folder and their evaluation
  GET    /files/...              files of the results folder (report.md, figures)

Service health and fault injection work with the e-commerce services only (profile ``ecommerce``);
with another profile (the POS) those lists are empty and the API (pos-api) has its own monitoring.

Manual faults are refused while an experiment runs, because a fault that is not in that
experiment's labels.csv would spoil its results.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .profiles import get_profile

STATIC_DIR = Path(__file__).parent / "static"
LOCK_NAME = ".experiment-running"
CHAOS_TARGETS = ("order-service", "inventory-service", "payment-service")

# Same presets as the experiment runner, so manual faults are comparable.
SEVERITY = {
    "LATENCY": {"low": {"latencyMs": 400}, "medium": {"latencyMs": 1200}, "high": {"latencyMs": 3000}},
    "ERROR": {"low": {"errorRate": 0.10}, "medium": {"errorRate": 0.30}, "high": {"errorRate": 0.60}},
    "DB_TIMEOUT": {"low": {"latencyMs": 1500, "errorRate": 0.20},
                   "medium": {"latencyMs": 3000, "errorRate": 0.40},
                   "high": {"latencyMs": 6000, "errorRate": 0.70}},
}


def chaos_targets() -> tuple[str, ...]:
    return CHAOS_TARGETS if get_profile().name == "ecommerce" else ()


def service_urls() -> dict[str, str]:
    if get_profile().name != "ecommerce":
        return {}
    return {
        "api-gateway": os.environ.get("GATEWAY_URL", "http://localhost:8080"),
        "order-service": os.environ.get("ORDER_SERVICE_URL", "http://localhost:8081"),
        "inventory-service": os.environ.get("INVENTORY_SERVICE_URL", "http://localhost:8082"),
        "payment-service": os.environ.get("PAYMENT_SERVICE_URL", "http://localhost:8083"),
    }


class FaultRequest(BaseModel):
    type: Literal["LATENCY", "ERROR", "DB_TIMEOUT"]
    severity: Literal["low", "medium", "high"] = "medium"
    durationSec: int = Field(90, ge=10, le=900)


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 3.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
        return json.loads(text) if text else {}


def register_dashboard(app: FastAPI, state, settings) -> None:
    urls = service_urls()
    targets = chaos_targets()
    results_root = Path(os.environ.get("RESULTS_ROOT", str(Path(settings.results_dir).parent)))

    def experiment_lock() -> dict | None:
        lock = results_root / LOCK_NAME
        if not lock.exists():
            return None
        try:
            return json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"out": "unknown"}

    @app.get("/", include_in_schema=False)
    def dashboard_page():
        return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")

    @app.get("/api/history")
    def history(minutes: float = 30.0) -> dict:
        with state.lock:
            points = list(state.history)
        if points:
            cutoff = points[-1]["t"] - max(1.0, minutes) * 60.0
            points = [p for p in points if p["t"] >= cutoff]
        return {"points": points}

    @app.get("/api/services")
    def services() -> dict:
        def probe(item):
            name, url = item
            info = {"name": name, "url": url, "health": "UNREACHABLE", "fault": None}
            try:
                info["health"] = _http_json("GET", url.rstrip("/") + "/actuator/health").get("status", "UNKNOWN")
            except urllib.error.HTTPError as e:
                info["health"] = "DOWN" if e.code == 503 else f"HTTP {e.code}"
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if name in targets and info["health"] != "UNREACHABLE":
                try:
                    status = _http_json("GET", url.rstrip("/") + "/admin/chaos")
                    info["fault"] = status if status.get("active") else None
                except (urllib.error.URLError, OSError, ValueError):
                    pass
            return info

        items = []
        if urls:
            with ThreadPoolExecutor(max_workers=len(urls)) as pool:
                items = list(pool.map(probe, urls.items()))
        return {"services": items, "experiment_running": experiment_lock(),
                "telegram": state.telegram is not None}

    def _chaos(service: str, method: str, body: dict | None = None) -> dict:
        if service not in targets:
            raise HTTPException(status_code=404, detail=(
                f"faults can be injected into {', '.join(targets)}" if targets
                else "fault injection is not available for this profile"))
        running = experiment_lock()
        if running:
            raise HTTPException(status_code=409, detail=(
                f"an experiment is running ({running.get('out', '?')}); manual faults would spoil its "
                f"results. Wait until it has finished."))
        try:
            return _http_json(method, urls[service].rstrip("/") + "/admin/chaos", body, timeout=10)
        except urllib.error.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"{service} answered HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise HTTPException(status_code=502, detail=f"{service} is not reachable: {e}") from None

    @app.post("/api/chaos/{service}")
    def inject(service: str, request: FaultRequest) -> dict:
        body = {"type": request.type, "durationSec": request.durationSec, **SEVERITY[request.type][request.severity]}
        return _chaos(service, "POST", body)

    @app.delete("/api/chaos/{service}")
    def clear(service: str) -> dict:
        return _chaos(service, "DELETE")

    @app.get("/api/experiments")
    def experiments() -> dict:
        running = experiment_lock()
        items = []
        if results_root.is_dir():
            for folder in sorted(p for p in results_root.iterdir() if (p / "labels.csv").exists()):
                meta = {}
                if (folder / "experiment.json").exists():
                    try:
                        meta = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        meta = {}
                is_running = bool(running) and Path(str(running.get("out", ""))).name == folder.name
                status = ("running" if is_running else
                          "incomplete" if not meta or meta.get("completed") is False else "completed")
                item = {"name": folder.name, "status": status,
                        "faults": sum(1 for _ in (folder / "labels.csv").open(encoding="utf-8")) - 1,
                        "seed": meta.get("seed"), "duration_min": meta.get("duration_min"),
                        "labels_path": str(folder / "labels.csv"), "evaluation": None}
                results_file = folder / "detector" / "results.json"
                if results_file.exists():
                    try:
                        res = json.loads(results_file.read_text(encoding="utf-8"))
                        item["evaluation"] = {
                            "out_dir": str(folder / "detector"),
                            "files": sorted(p.name for p in (folder / "detector").iterdir() if p.is_file()),
                            "metrics": res.get("metrics", {}),
                            "detection_delay": res.get("detection_delay", {}),
                            "comparison_vs_rules": res.get("comparison_vs_rules", {}),
                        }
                    except (OSError, ValueError):
                        pass
                items.append(item)
        return {"results_root": str(results_root), "experiment_running": running, "experiments": items}

    app.mount("/files", StaticFiles(directory=str(results_root), check_dir=False), name="files")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
