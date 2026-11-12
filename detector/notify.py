"""Telegram notifications for live detection.

Alerts are grouped into *incidents* so a fault does not produce a message every 10 seconds:
one message when the first alerting window appears ("anomaly detected", with the explanation)
and one when no method has alerted for ``resolve_after_s`` seconds ("resolved").

When Telegram cannot be reached, the message is kept and sent again later (with growing pauses);
a WARN is logged and the detector keeps running. Messages Telegram refuses for good (wrong token
or chat id) are logged and dropped.

Configuration (environment variables): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
NOTIFY_METHODS (comma-separated, default: the profile's notify methods), NOTIFY_RESOLVE_AFTER_S
(default 60), NOTIFY_TIMEZONE (time zone of the times in the messages, default UTC).
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import METHOD_LABELS

log = logging.getLogger("detector.notify")

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE = 4000          # Telegram's limit is 4096 characters


class TelegramError(RuntimeError):
    """A failed send. ``retry`` = worth trying again later (network, 5xx, 429)."""

    def __init__(self, message: str, retry: bool, retry_after: float | None = None):
        super().__init__(message)
        self.retry = retry
        self.retry_after = retry_after


class TelegramSender:
    """Sends plain-text messages to one Telegram chat through the Bot API.

    ``send`` delivers at once (used by the test endpoint). ``send_async`` queues the message for a
    background thread, which retries while Telegram is unreachable: after 10 s, 30 s, 1 min, 2 min,
    then every 5 min, for at most ``max_age_s`` (6 h). The queue keeps the newest ``max_queue`` messages.
    """

    RETRY_DELAYS = (10.0, 30.0, 60.0, 120.0, 300.0)

    def __init__(self, token: str, chat_id: str, timeout_s: float = 15.0, api_url: str = TELEGRAM_API,
                 max_age_s: float = 6 * 3600.0, max_queue: int = 100,
                 sleep: Callable[[float], None] | None = None):
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
        self.token = token
        self.chat_id = chat_id
        self.timeout_s = timeout_s
        self.api_url = api_url.rstrip("/")
        self.max_age_s = max_age_s
        self._queue: collections.deque[tuple[float, str]] = collections.deque(maxlen=max_queue)
        self._cond = threading.Condition()
        self._sleep = sleep
        self._worker: threading.Thread | None = None
        self._closed = False
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None

    def send(self, text: str) -> None:
        """Sends synchronously; raises TelegramError (a RuntimeError) with Telegram's reason on failure."""
        body = json.dumps({"chat_id": self.chat_id, "text": text[:MAX_MESSAGE],
                           "disable_web_page_preview": True}).encode("utf-8")
        request = urllib.request.Request(f"{self.api_url}/bot{self.token}/sendMessage", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            retry_after = _retry_after(detail)
            # 429 (too many requests) and 5xx are temporary; 400/401/403/404 (bad chat, token) are not
            raise TelegramError(f"Telegram returned HTTP {e.code}: {_telegram_reason(detail)}",
                                retry=e.code == 429 or e.code >= 500, retry_after=retry_after) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TelegramError(f"could not reach Telegram: {getattr(e, 'reason', e)}", retry=True) from None
        if not payload.get("ok", False):
            raise TelegramError(f"Telegram refused the message: {payload.get('description', payload)}", retry=False)

    def send_async(self, text: str) -> None:
        """Queues the message; a background thread sends it and retries while Telegram is down."""
        with self._cond:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
                log.warning("Telegram queue full: the oldest unsent message was dropped")
            self._queue.append((time.time(), text))
            if self._closed:
                return
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="telegram", daemon=True)
                self._worker.start()
            self._cond.notify()

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Waits until the queue is empty (tests); True when everything was sent or dropped."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.pending:
                return True
            time.sleep(0.02)
        return not self.pending

    def close(self) -> None:
        """Stops the background thread (unsent messages stay in the queue)."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _wait(self, seconds: float) -> None:
        if self._sleep is not None:
            self._sleep(seconds)
            return
        with self._cond:
            self._cond.wait(timeout=seconds)

    def _run(self) -> None:
        attempt = 0
        while True:
            with self._cond:
                if not self._queue or self._closed:
                    self._worker = None      # under the lock: send_async starts a new worker if needed
                    return
                queued_at, text = self._queue[0]
            if time.time() - queued_at > self.max_age_s:
                self._pop(text)
                self.dropped += 1
                log.warning("Telegram notification dropped: not delivered within %.0f h", self.max_age_s / 3600)
                attempt = 0
                continue
            try:
                self.send(text)
            except TelegramError as e:
                self.last_error = str(e)
                if not e.retry:
                    self._pop(text)
                    self.dropped += 1
                    log.warning("Telegram notification failed and will not be retried: %s", e)
                    attempt = 0
                    continue
                delay = e.retry_after or self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                attempt += 1
                log.warning("Telegram notification failed (%s); retrying in %.0f s, %d message(s) waiting",
                            e, delay, self.pending)
                self._wait(delay)
                continue
            self._pop(text)
            self.sent += 1
            self.last_error = None
            attempt = 0

    def _pop(self, text: str) -> None:
        with self._cond:
            if self._queue and self._queue[0][1] == text:
                self._queue.popleft()


def _retry_after(body: str) -> float | None:
    try:
        value = json.loads(body).get("parameters", {}).get("retry_after")
        return float(value) if value else None
    except (ValueError, AttributeError, TypeError):
        return None


def _telegram_reason(body: str) -> str:
    try:
        return json.loads(body).get("description", body)
    except (ValueError, AttributeError):
        return body


class IncidentNotifier:
    """Turns per-window detection results into 'detected' / 'resolved' notifications."""

    def __init__(self, send: Callable[[str], None], methods: Iterable[str],
                 resolve_after_s: float = 60.0, system_name: str = "", timezone: str = "UTC"):
        self.send = send
        self.methods = set(methods)
        self.resolve_after_s = resolve_after_s
        self.system_name = system_name
        self.timezone = timezone
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
                self.send(format_detected(result, firing, self.incidents, self.system_name, self.timezone))
                return "opened"
            self.methods_seen.update(firing)
            return None
        if self.open_since is not None and self.last_alert is not None \
                and now - self.last_alert >= self.resolve_after_s:
            message = format_resolved(self.incidents, self.open_since, self.last_alert, self.methods_seen,
                                      self.system_name, self.timezone)
            self.open_since = None
            self.last_alert = None
            self.methods_seen = set()
            self.send(message)
            return "resolved"
        return None


def _zone(name: str) -> dt.tzinfo:
    try:
        return ZoneInfo(name) if name and name.upper() != "UTC" else dt.timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("unknown NOTIFY_TIMEZONE %r; using UTC", name)
        return dt.timezone.utc


def _clock(epoch: float, timezone: str = "UTC") -> str:
    zone = _zone(timezone)
    label = "UTC" if zone is dt.timezone.utc else timezone
    return dt.datetime.fromtimestamp(epoch, tz=zone).strftime("%Y-%m-%d %H:%M:%S ") + label


def format_detected(result: dict, firing: list[str], incident: int, system_name: str = "",
                    timezone: str = "UTC") -> str:
    where = f" in the {system_name}" if system_name else ""
    lines = [f"🚨 Anomaly detected{where} (incident #{incident})",
             f"Time: {_clock(float(result['window_end_epoch']), timezone)}"]
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
        store = exp.get("store")
        if store:
            lines.append(f"Store: {store['store']} ({store['lines']} of {store['of']} failures)")
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


def format_resolved(incident: int, since: float, last_alert: float, methods: set[str],
                    system_name: str = "", timezone: str = "UTC") -> str:
    minutes = max(0.0, last_alert - since) / 60.0
    names = ", ".join(METHOD_LABELS.get(m, m) for m in sorted(methods))
    where = f" in the {system_name}" if system_name else ""
    return (f"✅ Resolved{where} (incident #{incident})\n"
            f"No alerts since {_clock(last_alert, timezone)}.\n"
            f"Alerting lasted about {minutes:.1f} min. Detected by: {names}")
