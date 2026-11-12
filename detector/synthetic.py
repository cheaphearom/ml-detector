"""Synthetic logs in exactly the JSON format the Spring Boot services write.

This exists only to test the detector without Docker. It simulates the same request mix,
saga flow, retries, timeouts and business noise (declined payments, the odd out-of-stock) and
injects faults whose intensity ramps up gradually over the first part of each fault, the way a
filling connection pool or a slowly degrading dependency would. **Do not use these logs for
thesis results** - they only exercise the code path end to end.

Output: ``<out>/logs/<service>.log``, ``<out>/labels.csv``, ``<out>/experiment.json``.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

SERVICES = ("api-gateway", "order-service", "inventory-service", "payment-service")
FAULT_TARGETS = ("order-service", "inventory-service", "payment-service")
FAULT_TYPES = ("LATENCY", "ERROR", "DB_TIMEOUT")
SEVERITY = {
    "LATENCY": {"low": {"latencyMs": 400}, "medium": {"latencyMs": 1200}, "high": {"latencyMs": 3000}},
    "ERROR": {"low": {"errorRate": 0.10}, "medium": {"errorRate": 0.30}, "high": {"errorRate": 0.60}},
    "DB_TIMEOUT": {"low": {"latencyMs": 1500, "errorRate": 0.20},
                   "medium": {"latencyMs": 3000, "errorRate": 0.40},
                   "high": {"latencyMs": 6000, "errorRate": 0.70}},
}
MIX = (("create_order", 0.55), ("get_order", 0.25), ("list_products", 0.12), ("list_accounts", 0.08))
ACTION_SERVICE = {"create_order": "order-service", "get_order": "order-service",
                  "list_products": "inventory-service", "list_accounts": "payment-service"}
ACTION_ROUTE = {"create_order": ("POST", "/api/orders", 202), "get_order": ("GET", "/api/orders/{id}", 200),
                "list_products": ("GET", "/api/products", 200), "list_accounts": ("GET", "/api/accounts", 200)}
ACTION_OPERATION = {"create_order": "createOrder", "get_order": "getOrder",
                    "list_products": "listProducts", "list_accounts": "listAccounts"}
PRODUCTS = [f"P-{i}" for i in range(100, 106)]
CUSTOMERS = [f"C-{i:03d}" for i in range(1, 21)]
POOR_CUSTOMERS = {"C-007", "C-014"}
GATEWAY_TIMEOUT_S = 10.0
SAGA_TIMEOUT_S = 15.0
SAGA_CHECK_S = 5.0

EXCEPTIONS = (
    ("org.springframework.dao.DataAccessResourceFailureException", "Unable to acquire JDBC Connection during {op}"),
    ("org.springframework.dao.CannotAcquireLockException",
     "could not obtain lock on row during {op}; SQL [select ... for update]"),
    ("org.springframework.dao.TransientDataAccessResourceException", "I/O error while sending to the backend during {op}"),
    ("java.lang.IllegalStateException", "Connection reset by peer while executing {op}"),
    ("java.io.UncheckedIOException", "java.net.SocketTimeoutException: Read timed out during {op}"),
    ("java.lang.NullPointerException",
     "Cannot invoke \"java.math.BigDecimal.compareTo(Object)\" because \"value\" is null in {op}"),
)
LOGGERS = {
    "access": "com.thesis.common.logging.AccessLogFilter",
    "gateway": "com.thesis.gateway.AccessLogGlobalFilter",
    "handler": "com.thesis.common.logging.GlobalExceptionHandler",
    "saga": "com.thesis.order.saga.SagaOrchestrator",
    "order_listener": "com.thesis.order.saga.SagaReplyListener",
    "order_kafka": "com.thesis.order.config.KafkaConfig",
    "inventory": "com.thesis.inventory.domain.InventoryService",
    "inventory_listener": "com.thesis.inventory.messaging.InventoryCommandListener",
    "inventory_kafka": "com.thesis.inventory.config.KafkaConfig",
    "payment": "com.thesis.payment.domain.PaymentService",
    "payment_listener": "com.thesis.payment.messaging.PaymentCommandListener",
    "payment_kafka": "com.thesis.payment.config.KafkaConfig",
}
LEVEL_VALUES = {"INFO": 20000, "WARN": 30000, "ERROR": 40000}


@dataclass
class Fault:
    fault_id: str
    service: str
    type: str
    start: float
    end: float
    severity: str
    params: dict
    ramp_frac: float

    def intensity(self, t: float) -> float:
        if not self.start <= t < self.end:
            return 0.0
        ramp = max(1.0, self.ramp_frac * (self.end - self.start))
        return min(1.0, (t - self.start) / ramp)


class _Sim:
    def __init__(self, rng: random.Random, faults: list[Fault]):
        self.rng = rng
        self.faults = faults
        self.records: dict[str, list[tuple[float, dict]]] = {s: [] for s in SERVICES}

    # -- helpers -----------------------------------------------------------------------
    def _hex(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789abcdef") for _ in range(n))

    def _uuid(self) -> str:
        h = self._hex(32)
        return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:]}"

    def emit(self, t: float, service: str, level: str, text: str, logger: str, fields: dict,
             trace: str, stack: str | None = None) -> None:
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        record = {
            "@timestamp": _iso(t), "@version": "1", "message": f"{text} {kv}".strip(),
            "logger_name": logger, "thread_name": f"{service}-worker-{self.rng.randint(1, 8)}",
            "level": level, "level_value": LEVEL_VALUES[level], "service": service,
            "traceId": trace, "spanId": self._hex(16), **fields,
        }
        if stack:
            record["stack_trace"] = stack
        self.records[service].append((t, record))

    def fault_effect(self, service: str, t: float, operation: str) -> tuple[float, tuple[str, str] | None]:
        """Added latency (s) and an optional (exception class, message) for an operation."""
        for f in self.faults:
            if f.service != service:
                continue
            k = f.intensity(t)
            if k <= 0:
                continue
            jitter = 0.75 + self.rng.random() * 0.5
            if f.type == "LATENCY":
                return f.params["latencyMs"] * k * jitter / 1000.0, None
            if f.type == "ERROR":
                if self.rng.random() < f.params["errorRate"] * k:
                    cls, msg = self.rng.choice(EXCEPTIONS)
                    return 0.0, (cls, msg.format(op=operation))
                return 0.0, None
            if f.type == "DB_TIMEOUT":
                waited = f.params["latencyMs"] * k * jitter
                if self.rng.random() < f.params["errorRate"] * k:
                    msg = (f"JDBC query timeout after {round(waited)}ms during {operation}; HikariPool-1 - "
                           f"Connection is not available, request timed out (Total=10, Active=10, Idle=0, "
                           f"Waiting={self.rng.randint(1, 40)})")
                    return waited / 1000.0, ("org.springframework.dao.QueryTimeoutException", msg)
                return waited / 1000.0, None
        return 0.0, None

    @staticmethod
    def _stack(exc: tuple[str, str], frames: list[str]) -> str:
        cls, msg = exc
        lines = [f"{cls}: {msg}"] + [f"\tat {f}" for f in frames]
        return "\n".join(lines)

    def _short(self, exc: tuple[str, str]) -> str:
        return exc[0].rsplit(".", 1)[-1]

    # -- HTTP --------------------------------------------------------------------------
    def access(self, t: float, service: str, method: str, path: str, status: int, dur_s: float, trace: str,
               gateway: bool = False) -> None:
        level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
        self.emit(t, service, level, "HTTP request", LOGGERS["gateway" if gateway else "access"],
                  {"event": "http_request", "method": method, "path": path, "status": status,
                   "durationMs": int(dur_s * 1000)}, trace)

    def request(self, t: float, action: str, customers_poor: set[str]) -> None:
        rng = self.rng
        service = ACTION_SERVICE[action]
        method, path, ok_status = ACTION_ROUTE[action]
        trace = self._hex(32)
        dur = rng.lognormvariate(math.log(0.012), 0.5)
        extra, exc = self.fault_effect(service, t, ACTION_OPERATION[action])
        dur += extra
        status = ok_status
        if exc:
            status = 500
            self.emit(t + dur, service, "ERROR", "Unhandled exception", LOGGERS["handler"],
                      {"event": "unhandled_exception", "path": path, "exception": self._short(exc),
                       "errorMessage": exc[1][:300]}, trace,
                      stack=self._stack(exc, [f"com.thesis.{service.split('-')[0]}.api.Controller.handle(Controller.java:{rng.randint(30, 80)})",
                                              "org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:885)"]))
        elif action == "get_order" and rng.random() < 0.01:
            status = 404
        self.access(t + dur, service, method, path, status, dur, trace)

        gw_dur = dur + rng.lognormvariate(math.log(0.003), 0.4)
        gw_status = status
        if gw_dur > GATEWAY_TIMEOUT_S:
            gw_dur, gw_status = GATEWAY_TIMEOUT_S, 504
        self.access(t + gw_dur, "api-gateway", method, path, gw_status, gw_dur, trace, gateway=True)

        if action == "create_order" and status == 202:
            customer = rng.choice(CUSTOMERS)
            self.saga(t + dur, trace, poor=customer in customers_poor, quantity=rng.randint(1, 3))

    def maintenance(self, t: float) -> None:
        for p in PRODUCTS:
            trace = self._hex(32)
            dur = self.rng.lognormvariate(math.log(0.015), 0.4)
            extra, exc = self.fault_effect("inventory-service", t, "restock")
            dur += extra
            status = 500 if exc else 200
            if not exc:
                self.emit(t + dur, "inventory-service", "INFO", "Product restocked", LOGGERS["inventory"],
                          {"event": "stock_restocked", "productId": p, "quantity": 150,
                           "available": self.rng.randint(300, 900)}, trace)
            self.access(t + dur, "inventory-service", "POST", "/api/products/{id}/restock", status, dur, trace)
            self.access(t + dur + 0.003, "api-gateway", "POST", "/api/products/{id}/restock", status, dur + 0.003,
                        trace, gateway=True)
            t += 0.05
        for c in CUSTOMERS:
            if c in POOR_CUSTOMERS:
                continue
            trace = self._hex(32)
            dur = self.rng.lognormvariate(math.log(0.012), 0.4)
            extra, exc = self.fault_effect("payment-service", t, "topUp")
            dur += extra
            status = 500 if exc else 200
            if not exc:
                self.emit(t + dur, "payment-service", "INFO", "Account topped up", LOGGERS["payment"],
                          {"event": "account_topped_up", "customerId": c, "amount": 5000}, trace)
            self.access(t + dur, "payment-service", "POST", "/api/accounts/{id}/topup", status, dur, trace)
            self.access(t + dur + 0.003, "api-gateway", "POST", "/api/accounts/{id}/topup", status, dur + 0.003,
                        trace, gateway=True)
            t += 0.05

    # -- Saga --------------------------------------------------------------------------
    def participant(self, t: float, service: str, topic: str, msg_type: str, operation: str, saga_id: str,
                    trace: str, business_fail: str | None) -> tuple[float, bool, str | None]:
        """Simulates a participant with 2 retries. Returns (reply time, success, reason)."""
        prefix = service.split("-")[0]
        listener, kafka_logger = LOGGERS[f"{prefix}_listener"], LOGGERS[f"{prefix}_kafka"]
        for attempt in range(3):
            dur = self.rng.lognormvariate(math.log(0.008), 0.5)
            extra, exc = self.fault_effect(service, t, operation)
            dur += extra
            if exc:
                self.emit(t + dur, service, "ERROR", "Message processing failed", listener,
                          {"event": "message_failed", "topic": topic, "messageType": msg_type, "sagaId": saga_id,
                           "durationMs": int(dur * 1000), "exception": self._short(exc)}, trace,
                          stack=self._stack(exc, [f"com.thesis.{prefix}.domain.{prefix.capitalize()}Service.handle({prefix.capitalize()}Service.java:{self.rng.randint(40, 120)})",
                                                  f"com.thesis.{prefix}.messaging.{prefix.capitalize()}CommandListener.onCommand"]))
                t += dur + 1.0
                continue
            success = business_fail is None
            if not success:
                event = "stock_rejected" if prefix == "inventory" else "payment_declined"
                self.emit(t + dur * 0.9, service, "INFO",
                          "Stock reservation rejected" if prefix == "inventory" else "Payment declined",
                          LOGGERS[prefix], {"event": event, "sagaId": saga_id, "reason": business_fail}, trace)
            else:
                event = {"RESERVE": "stock_reserved", "RELEASE": "stock_released", "CHARGE": "payment_charged",
                         "REFUND": "payment_refunded"}[msg_type]
                self.emit(t + dur * 0.9, service, "INFO", event.replace("_", " ").capitalize(), LOGGERS[prefix],
                          {"event": event, "sagaId": saga_id}, trace)
            self.emit(t + dur, service, "INFO", "Message processed", listener,
                      {"event": "message_processed", "topic": topic, "messageType": msg_type, "sagaId": saga_id,
                       "success": str(success).lower(), "durationMs": int(dur * 1000)}, trace)
            return t + dur + self.rng.uniform(0.002, 0.008), success, business_fail
        self.emit(t, service, "ERROR", "Message retries exhausted", kafka_logger,
                  {"event": "message_retries_exhausted", "topic": topic, "messageType": msg_type, "sagaId": saga_id},
                  trace)
        return t + 0.005, False, "PROCESSING_ERROR"

    def order_handle(self, t: float, topic: str, msg_type: str, saga_id: str, trace: str,
                     operation: str) -> float | None:
        """Order-service reply handling with 3 retries. Returns completion time or None if dropped."""
        for attempt in range(4):
            dur = self.rng.lognormvariate(math.log(0.006), 0.5)
            extra, exc = self.fault_effect("order-service", t, operation)
            dur += extra
            if exc:
                self.emit(t + dur, "order-service", "ERROR", "Message processing failed", LOGGERS["order_listener"],
                          {"event": "message_failed", "topic": topic, "messageType": msg_type, "sagaId": saga_id,
                           "durationMs": int(dur * 1000), "exception": self._short(exc)}, trace,
                          stack=self._stack(exc, ["com.thesis.order.saga.SagaOrchestrator.handleReply",
                                                  "com.thesis.order.saga.SagaReplyListener.onReply"]))
                t += dur + 1.0
                continue
            self.emit(t + dur, "order-service", "INFO", "Message processed", LOGGERS["order_listener"],
                      {"event": "message_processed", "topic": topic, "messageType": msg_type, "sagaId": saga_id,
                       "durationMs": int(dur * 1000)}, trace)
            return t + dur
        self.emit(t, "order-service", "WARN", "Message dropped after retries", LOGGERS["order_kafka"],
                  {"event": "message_dropped", "topic": topic, "partition": self.rng.randint(0, 2),
                   "offset": self.rng.randint(1000, 99999)}, trace)
        return None

    def saga(self, t0: float, trace: str, poor: bool, quantity: int) -> None:
        rng = self.rng
        saga_id, order_id = self._uuid(), self._uuid()
        self.emit(t0, "order-service", "INFO", "Saga started", LOGGERS["saga"],
                  {"event": "saga_started", "sagaId": saga_id, "orderId": order_id,
                   "productId": rng.choice(PRODUCTS), "quantity": quantity}, trace)
        state = {"last": t0, "done": False}

        def timed_out(t_reply: float) -> bool:
            """Emits saga_timeout if the scheduler would have fired before t_reply."""
            deadline = state["last"] + SAGA_TIMEOUT_S
            if t_reply is not None and t_reply <= deadline:
                return False
            fire = deadline + rng.uniform(0, SAGA_CHECK_S)
            self.emit(fire, "order-service", "WARN", "Saga timed out", LOGGERS["saga"],
                      {"event": "saga_timeout", "sagaId": saga_id, "orderId": order_id,
                       "durationMs": int((fire - t0) * 1000)}, trace)
            if t_reply is not None:
                self.emit(max(t_reply, fire + 0.01), "order-service", "WARN", "Late reply for finished saga",
                          LOGGERS["saga"], {"event": "saga_late_reply", "sagaId": saga_id, "orderId": order_id,
                                            "success": "true"}, trace)
            state["done"] = True
            return True

        def outcome(t: float, event: str, reason: str) -> None:
            business = reason in ("OUT_OF_STOCK", "INSUFFICIENT_FUNDS")
            text = {"saga_failed": "Saga failed", "saga_compensated": "Saga compensated"}[event]
            self.emit(t, "order-service", "INFO" if business else "WARN", text, LOGGERS["saga"],
                      {"event": event, "sagaId": saga_id, "orderId": order_id, "reason": reason,
                       "durationMs": int((t - t0) * 1000)}, trace)

        # RESERVE
        stock_fail = "OUT_OF_STOCK" if rng.random() < 0.01 else None
        t_reply, ok, reason = self.participant(t0 + 0.005, "inventory-service", "inventory-commands", "RESERVE",
                                               "reserveStock", saga_id, trace, stock_fail)
        if timed_out(t_reply):
            return
        t_done = self.order_handle(t_reply, "inventory-replies", "RESERVE", saga_id, trace, "handleInventoryReply")
        if t_done is None:
            timed_out(None)
            return
        if not ok:
            outcome(t_done, "saga_failed", reason)
            return
        state["last"] = t_done
        self.emit(t_done, "order-service", "INFO", "Saga step", LOGGERS["saga"],
                  {"event": "saga_step", "sagaId": saga_id, "orderId": order_id, "step": "PROCESS_PAYMENT"}, trace)

        # CHARGE
        decline = "INSUFFICIENT_FUNDS" if poor and rng.random() < 0.85 else None
        t_reply, ok, reason = self.participant(t_done + 0.005, "payment-service", "payment-commands", "CHARGE",
                                               "chargePayment", saga_id, trace, decline)
        if timed_out(t_reply):
            return
        t_done = self.order_handle(t_reply, "payment-replies", "CHARGE", saga_id, trace, "handlePaymentReply")
        if t_done is None:
            timed_out(None)
            return
        if ok:
            self.emit(t_done, "order-service", "INFO", "Saga completed", LOGGERS["saga"],
                      {"event": "saga_completed", "sagaId": saga_id, "orderId": order_id,
                       "durationMs": int((t_done - t0) * 1000)}, trace)
            return
        state["last"] = t_done
        self.emit(t_done, "order-service", "INFO" if reason == "INSUFFICIENT_FUNDS" else "WARN", "Saga compensating",
                  LOGGERS["saga"], {"event": "saga_compensating", "sagaId": saga_id, "orderId": order_id,
                                    "reason": reason}, trace)
        # RELEASE (compensation)
        t_reply, _, _ = self.participant(t_done + 0.005, "inventory-service", "inventory-commands", "RELEASE",
                                         "releaseStock", saga_id, trace, None)
        if timed_out(t_reply):
            return
        t_done = self.order_handle(t_reply, "inventory-replies", "RELEASE", saga_id, trace, "handleInventoryReply")
        if t_done is None:
            timed_out(None)
            return
        outcome(t_done, "saga_compensated", reason)


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def schedule_faults(rng: random.Random, t_start: float, duration_s: float, warmup_s: float,
                    min_fault_s: int, max_fault_s: int, min_gap_s: int, max_gap_s: int,
                    ramp_frac: float, types: tuple[str, ...] = FAULT_TYPES) -> list[Fault]:
    faults = []
    t = t_start + warmup_s
    end = t_start + duration_s
    n = 0
    while True:
        service = rng.choice(FAULT_TARGETS)
        ftype = rng.choice(types)
        severity = rng.choice(("low", "medium", "high"))
        dur = rng.randint(min_fault_s, max_fault_s)
        if t + dur + min_gap_s > end:
            break
        n += 1
        faults.append(Fault(f"F{n:03d}", service, ftype, t, t + dur, severity, dict(SEVERITY[ftype][severity]),
                            ramp_frac))
        t += dur + rng.randint(min_gap_s, max_gap_s)
    return faults


def generate(out_dir: str | Path, duration_min: float = 60.0, warmup_min: float = 5.0, rps: float = 5.0,
             seed: int = 1, min_fault_s: int = 60, max_fault_s: int = 180, min_gap_s: int = 180,
             max_gap_s: int = 420, start: dt.datetime | None = None, ramp_frac: float = 0.4,
             types: tuple[str, ...] = FAULT_TYPES) -> dict:
    """Writes synthetic logs and labels to ``out_dir`` and returns the experiment metadata."""
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    start = start or dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    t_start = start.timestamp()
    duration_s = duration_min * 60.0
    faults = schedule_faults(rng, t_start, duration_s, warmup_min * 60.0, min_fault_s, max_fault_s,
                             min_gap_s, max_gap_s, ramp_frac, types)
    sim = _Sim(rng, faults)

    t = t_start
    next_maintenance = t_start + 120.0
    actions = [a for a, _ in MIX]
    weights = [w for _, w in MIX]
    t_end = t_start + duration_s
    while True:
        t += rng.expovariate(rps)
        if t >= t_end:
            break
        while next_maintenance <= t:
            sim.maintenance(next_maintenance)
            next_maintenance += 120.0
        sim.request(t, rng.choices(actions, weights)[0], POOR_CUSTOMERS)

    for service, records in sim.records.items():
        records.sort(key=lambda r: r[0])
        with (out / "logs" / f"{service}.log").open("w", encoding="utf-8") as fh:
            for ts, record in records:
                if ts <= t_end + 30:
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fault_id", "service", "type", "start", "end", "params"])
        writer.writeheader()
        for f in faults:
            writer.writerow({"fault_id": f.fault_id, "service": f.service, "type": f.type,
                             "start": _iso(f.start), "end": _iso(f.end),
                             "params": json.dumps({"severity": f.severity, "durationSec": round(f.end - f.start),
                                                   **f.params}, separators=(",", ":"))})
    meta = {
        "synthetic": True, "seed": seed, "rps": rps, "duration_min": duration_min, "warmup_min": warmup_min,
        "fault_duration_s": [min_fault_s, max_fault_s], "gap_s": [min_gap_s, max_gap_s],
        "ramp_frac": ramp_frac, "load_start": _iso(t_start), "warmup_end": _iso(t_start + warmup_min * 60),
        "load_end": _iso(t_end), "faults": len(faults),
    }
    (out / "experiment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
