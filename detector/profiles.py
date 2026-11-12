"""Detector profiles: which system the detector watches.

The detector was written for the e-commerce saga system. A profile holds everything that is
specific to one system, so the same code can watch another one (the POS):

    services          services whose logs are turned into per-service features
    entry_service     service the rules check for 5xx ratio and p95 latency (the front door)
    business_events   event names counted per window as system features (``system__<event>``)
    success_ratio     ``system__<name>`` = sum(numerator) / sum(denominator), 1.0 when nothing happened
    symptom_events    business events that indicate a problem; used by Isolation Forest, PCA and the
                      explanation (counts of normal activity, such as completed sales, are left out)
    rules             default rule thresholds (see ``models.RuleConfig``)
    event_limits      extra rules: more than N lines of an event in one window
    pca_t2            PCA also scores Hotelling's T² (see ``models.PCADetector``)
    live_methods      methods the live detector runs (supervised ones only when labels exist)
    notify_methods    methods whose alerts are sent to Telegram (``NOTIFY_METHODS`` overrides it)
    notify_min_windows  an ML method notifies only after alerting in this many windows in a row (the rules have
                      their own "consecutive"); 4 windows ≈ 40 s, so one stray error does not page anyone
    system_name       shown in notifications ("Anomaly detected in the POS"); empty for the thesis system
    kind_texts        wording of the latency / error kinds of problem (default: the thesis fault names)
    event_problems    [event, problem, service]: an event far above normal names this problem and, when no
                      service stands out by itself, this service

Choose one with ``DETECTOR_PROFILE=ecommerce|pos`` (default ``ecommerce``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

PROFILES: dict[str, dict] = {
    "ecommerce": {
        "description": "E-commerce saga system (thesis): gateway, order, inventory and payment services",
        "services": ["api-gateway", "order-service", "inventory-service", "payment-service"],
        "entry_service": "api-gateway",
        "business_events": ["saga_started", "saga_completed", "saga_failed", "saga_compensating",
                            "saga_compensated", "saga_timeout", "saga_late_reply"],
        "success_ratio": {"name": "saga_success_ratio", "numerator": ["saga_completed"],
                          "denominator": ["saga_started"]},
        "symptom_events": ["saga_timeout", "saga_late_reply"],
        "rules": {"gateway_5xx_ratio": 0.05, "gateway_p95_ms": 2000.0, "errors_per_service": 10.0,
                  "min_requests": 10, "consecutive": 3},
        "event_limits": {},
        "pca_t2": False,
        "live_methods": ["rules", "isolation_forest", "random_forest"],
        "notify_methods": ["rules", "isolation_forest", "random_forest"],
        "notify_min_windows": 1,
        "system_name": "",
        "kind_texts": {},
        "event_problems": [],
        "readable": {
            "saga_timeout": "saga timeouts",
            "saga_late_reply": "late saga replies",
            "saga_success_ratio": "saga success ratio",
            "kafka_p95": "Kafka message p95 (ms)",
        },
        "notes": {"saga_timeout": "orders are getting stuck (saga timeouts)"},
    },
    "pos": {
        "description": "POS system: NestJS API (pos-api) and its BullMQ webhook worker (pos-worker)",
        "services": ["pos-api", "pos-worker"],
        "entry_service": "pos-api",
        "business_events": ["sale_completed", "sale_failed", "payment_failed", "stock_negative", "cash_mismatch",
                            "webhook_failed", "webhook_retries_exhausted", "login_failed", "db_query_slow"],
        "success_ratio": {"name": "sale_success_ratio", "numerator": ["sale_completed"],
                          "denominator": ["sale_completed", "sale_failed", "payment_failed"]},
        "symptom_events": ["sale_failed", "payment_failed", "webhook_failed", "webhook_retries_exhausted",
                           "login_failed", "db_query_slow"],
        "rules": {"gateway_5xx_ratio": 0.05, "gateway_p95_ms": 1500.0, "errors_per_service": 5.0,
                  "min_requests": 10, "consecutive": 3},
        # more than N in one window; 0 = any
        "event_limits": {"payment_failed": 3, "webhook_retries_exhausted": 0},
        # Few features change in a POS fault (e.g. only pos-api latency): SPE alone misses that.
        "pca_t2": True,
        # No labelled faults in the POS: unsupervised models plus the rules. Random Forest only
        # runs when a labels file exists. Isolation Forest is scored but does not notify: with few
        # features moving in a POS fault it misses most of them (see ORIGIN.md).
        "live_methods": ["rules", "isolation_forest", "pca", "random_forest"],
        "notify_methods": ["rules", "pca"],
        # One stray 500 stays in 3 overlapping windows; a real problem lasts longer.
        "notify_min_windows": 4,
        "system_name": "POS",
        "kind_texts": {
            "both": "slow responses and errors (database or connection problem)",
            "latency": "slow responses",
            "errors": "errors (HTTP 5xx / exceptions)",
        },
        "event_problems": [
            ["webhook_retries_exhausted", "webhook delivery failures", "pos-worker"],
            ["webhook_failed", "webhook delivery failures", "pos-worker"],
            ["db_query_slow", "slow database", "pos-api"],
            ["payment_failed", "payment failures", "pos-api"],
            ["sale_failed", "failed sales", "pos-api"],
            ["login_failed", "many failed logins", "pos-api"],
        ],
        "readable": {
            "sale_failed": "failed sales",
            "payment_failed": "failed payments",
            "webhook_failed": "failed webhook deliveries",
            "webhook_retries_exhausted": "webhooks given up after all retries",
            "login_failed": "failed logins",
            "db_query_slow": "slow database queries",
            "sale_success_ratio": "sale success ratio",
            "kafka_p95": "queue job p95 (ms)",
        },
        "notes": {
            "webhook_retries_exhausted": "webhook deliveries were given up after all retries",
            "db_query_slow": "the database is answering slowly",
        },
    },
}

DEFAULT_PROFILE = "ecommerce"


@dataclass(frozen=True)
class SuccessRatio:
    name: str
    numerator: tuple[str, ...]
    denominator: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    name: str
    services: tuple[str, ...]
    entry_service: str
    business_events: tuple[str, ...]
    success_ratio: SuccessRatio
    symptom_events: tuple[str, ...]
    rules: dict = field(default_factory=dict, hash=False, compare=False)
    event_limits: tuple[tuple[str, float], ...] = ()
    pca_t2: bool = False
    notify_min_windows: int = 1
    system_name: str = ""
    kind_texts: dict = field(default_factory=dict, hash=False, compare=False)
    event_problems: tuple[tuple[str, str, str], ...] = ()
    live_methods: tuple[str, ...] = ("rules", "isolation_forest", "random_forest")
    notify_methods: tuple[str, ...] = ("rules", "isolation_forest", "random_forest")
    readable: dict = field(default_factory=dict, hash=False, compare=False)
    notes: dict = field(default_factory=dict, hash=False, compare=False)
    description: str = ""

    @property
    def system_features(self) -> tuple[str, ...]:
        """Names (without ``system__``) of the system-wide features, in column order."""
        return self.business_events + (self.success_ratio.name, "unseen_templates")

    @property
    def symptom_system_features(self) -> tuple[str, ...]:
        """System features that indicate a problem (used by the unsupervised models)."""
        return self.symptom_events + (self.success_ratio.name, "unseen_templates")

    @property
    def backends(self) -> tuple[str, ...]:
        """Services behind the entry service; with two services or fewer every service counts."""
        rest = tuple(s for s in self.services if s != self.entry_service)
        return rest if len(self.services) > 2 else self.services


def _build(name: str, spec: dict) -> Profile:
    services = tuple(spec["services"])
    if spec["entry_service"] not in services:
        raise ValueError(f"profile {name}: entry_service {spec['entry_service']!r} is not one of {services}")
    events = tuple(spec["business_events"])
    ratio = spec["success_ratio"]
    unknown = set(spec.get("symptom_events", ())) - set(events)
    if unknown:
        raise ValueError(f"profile {name}: symptom_events {sorted(unknown)} are not business_events")
    return Profile(
        name=name, services=services, entry_service=spec["entry_service"], business_events=events,
        success_ratio=SuccessRatio(ratio["name"], tuple(ratio["numerator"]), tuple(ratio["denominator"])),
        symptom_events=tuple(spec.get("symptom_events", ())), rules=dict(spec.get("rules", {})),
        event_limits=tuple((k, float(v)) for k, v in spec.get("event_limits", {}).items()),
        pca_t2=bool(spec.get("pca_t2", False)),
        notify_min_windows=int(spec.get("notify_min_windows", 1)),
        system_name=spec.get("system_name", ""),
        kind_texts=dict(spec.get("kind_texts", {})),
        event_problems=tuple((e, text, svc) for e, text, svc in spec.get("event_problems", ())),
        live_methods=tuple(spec.get("live_methods", Profile.live_methods)),
        notify_methods=tuple(spec.get("notify_methods", spec.get("live_methods", Profile.notify_methods))),
        readable=dict(spec.get("readable", {})), notes=dict(spec.get("notes", {})),
        description=spec.get("description", ""),
    )


_CACHE: dict[str, Profile] = {}


def get_profile(name: str | Profile | None = None) -> Profile:
    """The named profile; without a name the one in ``DETECTOR_PROFILE`` (default ``ecommerce``)."""
    if isinstance(name, Profile):
        return name
    name = (name or os.environ.get("DETECTOR_PROFILE") or DEFAULT_PROFILE).strip().lower()
    if name not in PROFILES:
        raise ValueError(f"unknown DETECTOR_PROFILE {name!r}; choose one of {', '.join(PROFILES)}")
    if name not in _CACHE:
        _CACHE[name] = _build(name, PROFILES[name])
    return _CACHE[name]
