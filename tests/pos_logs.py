"""Synthetic POS logs (pos-api + pos-worker) in the format the NestJS API writes."""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

START = dt.datetime(2026, 11, 12, 6, 0, tzinfo=dt.timezone.utc)


def _line(t: float, service: str, level: str, logger: str, text: str, **fields) -> str:
    ts = (START + dt.timedelta(seconds=t)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    kv = " ".join(f"{k}={v}" for k, v in fields.items() if k != "stack_trace")
    return json.dumps({"level": level, "@timestamp": ts, "service": service, "traceId": f"{random.getrandbits(128):032x}",
                       "logger_name": logger, **fields, "message": f"{text} {kv}".strip()})


def pos_lines(duration_s: float, seed: int = 1, fault: tuple[float, float, str] | None = None) -> list[str]:
    """Normal till traffic; ``fault`` = (start, end, kind) with kind LATENCY, ERROR, WEBHOOK,
    PAYMENT (payments fail, 90 % at the Siem Reap store) or SLOWDB (slow queries, slower requests)."""
    rng = random.Random(seed)
    random.seed(seed)
    out: list[tuple[float, str]] = []
    t = 0.0
    while t < duration_s:
        t += rng.expovariate(2.0)                       # ~2 requests per second
        in_fault = fault is not None and fault[0] <= t < fault[1]
        kind = fault[2] if in_fault else None
        path, method = rng.choice([("/api/v1/products", "GET"), ("/api/v1/products/barcode/{id}", "GET"),
                                   ("/api/v1/sales", "POST"), ("/api/v1/reports/dashboard", "GET")])
        dur = rng.randint(8, 60) + (rng.randint(1800, 3500) if kind == "LATENCY" else 0)
        if kind == "SLOWDB":
            q = rng.randint(700, 1400)
            dur += q
            out.append((t + q / 1000, _line(t + q / 1000, "pos-api", "WARN", "PrismaQuery", "Slow database query",
                                            event="db_query_slow", model="sales", action="SELECT", durationMs=q)))
        status = 201 if method == "POST" else 200
        if kind == "ERROR" and rng.random() < 0.4:
            status = 500
            out.append((t, _line(t, "pos-api", "ERROR", "GlobalExceptionFilter", "Unhandled exception",
                                 event="unhandled_exception", method=method, path=path, exception="PrismaClientKnownRequestError",
                                 errorMessage="Timed out fetching a new connection from the connection pool",
                                 stack_trace="PrismaClientKnownRequestError: Timed out\n    at Prisma")))
        elif rng.random() < 0.03:
            status = 404
        if path == "/api/v1/sales" and status < 400 and kind == "PAYMENT" and rng.random() < 0.8:
            status = 400
            store = ("s2", "Siem Reap - Pub Street") if rng.random() < 0.9 else ("s1", "Phnom Penh - Central")
            out.append((t, _line(t, "pos-api", "WARN", "BusinessEvent", "Payment failed", event="payment_failed",
                                 storeId=store[0], storeName=store[1], paymentMethod="KHQR", status=400,
                                 reason="This KHQR has not been paid yet")))
        elif path == "/api/v1/sales" and status < 400:
            if rng.random() < 0.05:
                status = 400
                out.append((t, _line(t, "pos-api", "WARN", "BusinessEvent", "Sale failed", event="sale_failed",
                                     storeId="s1", status=400, reason="Not enough stock")))
            else:
                out.append((t, _line(t, "pos-api", "INFO", "BusinessEvent", "Sale completed", event="sale_completed",
                                     storeId="s1", storeName="Phnom Penh - Central", amount=4.5, durationMs=dur)))
                w = t + 0.05
                if kind == "WEBHOOK":
                    out.append((w, _line(w, "pos-worker", "WARN", "WebhooksProcessor", "Message failed",
                                         event="message_failed", queue="webhooks", durationMs=10000, errorMessage="fetch failed")))
                    out.append((w, _line(w, "pos-worker", "WARN", "WebhooksProcessor", "Webhook delivery failed",
                                         event="webhook_failed", queue="webhooks", durationMs=10000)))
                    out.append((w, _line(w, "pos-worker", "WARN", "WebhooksProcessor", "Webhook retries exhausted",
                                         event="webhook_retries_exhausted", queue="webhooks")))
                else:
                    wd = rng.randint(20, 80)
                    out.append((w, _line(w, "pos-worker", "INFO", "WebhooksProcessor", "Message processed",
                                         event="message_processed", queue="webhooks", durationMs=wd)))
                    out.append((w, _line(w, "pos-worker", "INFO", "WebhooksProcessor", "Webhook delivered",
                                         event="webhook_delivered", queue="webhooks", durationMs=wd)))
        level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
        out.append((t + dur / 1000, _line(t + dur / 1000, "pos-api", level, "HttpAccess", "HTTP request",
                                          event="http_request", method=method, path=path, status=status, durationMs=dur)))
    return [line for _, line in sorted(out, key=lambda p: p[0])]


def write_pos_logs(folder: Path, duration_s: float, seed: int = 1, fault=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pos-api.log").write_text("\n".join(pos_lines(duration_s, seed, fault)) + "\n", encoding="utf-8")
    return folder
