"""Telegram notifications for live detection.

Alerts are grouped into *incidents* so a fault does not produce a message every 10 seconds:
one message when the first alerting window appears ("anomaly detected", with the explanation)
and one when no method has alerted for ``resolve_after_s`` seconds ("resolved").

Configuration (environment variables): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
NOTIFY_METHODS (comma-separated, default: all three), NOTIFY_RESOLVE_AFTER_S (default 60).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable

from .models import METHOD_LABELS

log = logging.getLogger("detector.notify")

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE = 4000          # Telegram's limit is 4096 characters


class TelegramSender:
    """Sends plain-text messages to one Telegram chat through the Bot API."""

    def __init__(self, token: str, chat_id: str, timeout_s: float = 15.0):
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
        self.token = token
        self.chat_id = chat_id
        self.timeout_s = timeout_s

    def send(self, text: str) -> None:
        """Sends synchronously; raises RuntimeError with Telegram's reason on failure."""
        body = json.dumps({"chat_id": self.chat_id, "text": text[:MAX_MESSAGE],
                           "disable_web_page_preview": True}).encode("utf-8")
        request = urllib.request.Request(f"{TELEGRAM_API}/bot{self.token}/sendMessage", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Telegram returned HTTP {e.code}: {_telegram_reason(detail)}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"could not reach Telegram: {e}") from None
        if not payload.get("ok", False):
            raise RuntimeError(f"Telegram refused the message: {payload.get('description', payload)}")

    def send_async(self, text: str) -> None:
        """Sends in a background thread so the detection loop is never blocked."""
        def run():
            try:
                self.send(text)
            except RuntimeError as e:
                log.warning("Telegram notification failed: %s", e)
        threading.Thread(target=run, name="telegram", daemon=True).start()


def _telegram_reason(body: str) -> str:
    try:
        return json.loads(body).get("description", body)
    except (ValueError, AttributeError):
        return body


class IncidentNotifier:
    """Turns per-window detection results into 'detected' / 'resolved' notifications."""

    def __init__(self, send: Callable[[str], None], methods: Iterable[str],
                 resolve_after_s: float = 60.0):
        self.send = send
        self.methods = set(methods)
        self.resolve_after_s = resolve_after_s
        self.open_since: float | None = None
        self.last_alert: float | None = None
        self.methods_seen: set[str] = set()
        self.incidents = 0

    def update(self, result: dict) -> str | None:
        """Feeds one detection result; returns 'opened', 'resolved' or None."""
        now = float(result["window_end_epoch"])
        firing = [m for m, info in result["methods"].items() if info["alert"] and m in self.methods]
        if firing:
            self.last_alert = now
            if self.open_since is None:
                self.open_since = now
                self.methods_seen = set(firing)
                self.incidents += 1
                self.send(format_detected(result, firing, self.incidents))
                return "opened"
            self.methods_seen.update(firing)
            return None
        if self.open_since is not None and self.last_alert is not None \
                and now - self.last_alert >= self.resolve_after_s:
            message = format_resolved(self.incidents, self.open_since, self.last_alert, self.methods_seen)
            self.open_since = None
            self.last_alert = None
            self.methods_seen = set()
            self.send(message)
            return "resolved"
        return None


def _clock(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_detected(result: dict, firing: list[str], incident: int) -> str:
    lines = [f"🚨 Anomaly detected (incident #{incident})",
             f"Time: {_clock(float(result['window_end_epoch']))}"]
    detected_by = ", ".join(f"{METHOD_LABELS.get(m, m)} ({result['methods'][m]['score']:.2f})" for m in firing)
    lines.append(f"Detected by: {detected_by}")
    rules = result["methods"].get("rules", {}).get("fired_rules")
    if rules:
        lines.append(f"Rules fired: {', '.join(rules)}")

    exp = result.get("explanation") or {}
    if exp:
        lines.append("")
        lines.append(f"Suspected service: {exp.get('suspected_service') or 'unclear'}")
        lines.append(f"Suspected problem: {exp.get('suspected_problem', 'unknown')}")
        if exp.get("top_signals"):
            lines.append("")
            lines.append("Strongest signals:")
            lines += [f"• {s['text']}" for s in exp["top_signals"]]
        if exp.get("notes"):
            lines.append("")
            lines += [f"ℹ️ {n}" for n in exp["notes"]]
        if exp.get("example_errors"):
            lines.append("")
            lines.append("Example log lines:")
            lines += [f"• {e}" for e in exp["example_errors"]]
    return "\n".join(lines)


def format_resolved(incident: int, since: float, last_alert: float, methods: set[str]) -> str:
    minutes = max(0.0, last_alert - since) / 60.0
    names = ", ".join(METHOD_LABELS.get(m, m) for m in sorted(methods))
    return (f"✅ Resolved (incident #{incident})\n"
            f"No alerts since {_clock(last_alert)}.\n"
            f"Alerting lasted about {minutes:.1f} min. Detected by: {names}")
