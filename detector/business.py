"""Business alerts: products that ran out of stock in a store.

A sale refused with "Not enough stock for Dish Soap 500ml (SKU-0022): available 0, needed 2" is not a
software fault, so the anomaly detection leaves it out (see ``parsing.STOCK_OUT_EVENT``). The shop owner still
wants to know: a cashier could not sell something a customer wanted. :class:`StockOutWatch` collects these
refused sales from the live log stream and sends one short Telegram message per store, naming the products.
The same product in the same store is reported again only after ``cooldown_s`` (default 60 min).

Configuration (environment variables): STOCK_ALERTS (on), STOCK_ALERT_COOLDOWN_MIN (60).
"""
from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .parsing import STOCK_OUT_EVENT, STOCK_OUT_REASON


@dataclass
class StockOut:
    store_id: str
    store: str
    product: str
    sku: str
    available: float
    needed: float
    t: float            # the latest refused sale (epoch seconds)
    refused: int = 1    # refused sales since the last message about this product

    def view(self) -> dict:
        return {"store_id": self.store_id, "store": self.store, "product": self.product, "sku": self.sku,
                "available": self.available, "needed": self.needed, "refused_sales": self.refused, "t": self.t}


def _num(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _qty(value: float) -> str:
    return f"{value:g}"


def stock_outs(logs: pd.DataFrame) -> list[StockOut]:
    """The refused sales in ``logs``, one entry per (store, product), counting the refusals."""
    if logs.empty or "detail" not in logs.columns:
        return []
    rows = logs[logs["event"] == STOCK_OUT_EVENT]
    found: dict[tuple[str, str], StockOut] = {}
    for r in rows.itertuples(index=False):
        m = STOCK_OUT_REASON.match(str(r.detail))
        if not m:
            continue
        store_id = str(r.store_id or "")
        key = (store_id, m["sku"] or m["product"])
        item = found.get(key)
        if item is None:
            found[key] = StockOut(store_id, str(r.store_name or "") or store_id or "unknown store", m["product"],
                                  m["sku"], _num(m["available"]), _num(m["needed"]), float(r.t))
        else:   # the newest refusal tells the current stock
            item.refused += 1
            if float(r.t) >= item.t:
                item.t, item.available, item.needed = float(r.t), _num(m["available"]), _num(m["needed"])
    return list(found.values())


def format_stock_out(store: str, items: list[StockOut], clock: Callable[[float], str], system_name: str = "") -> str:
    where = f" ({system_name})" if system_name else ""
    lines = [f"🛒 Out of stock: {store}{where}",
             f"Time: {clock(max(i.t for i in items))}", ""]
    for i in sorted(items, key=lambda i: i.product.lower()):
        sku = f" ({i.sku})" if i.sku else ""
        times = f", {i.refused} sales refused" if i.refused > 1 else ", 1 sale refused"
        lines.append(f"• {i.product}{sku}: {_qty(i.available)} left, a customer wanted {_qty(i.needed)}{times}")
    lines += ["", "Please restock or record the delivery in Inventory. This is a business alert, not a system fault."]
    return "\n".join(lines)


class StockOutWatch:
    """Turns refused sales in new log lines into at most one message per (store, product) per ``cooldown_s``."""

    def __init__(self, send: Callable[[str], None] | None, clock: Callable[[float], str],
                 cooldown_s: float = 3600.0, system_name: str = "", keep: int = 200):
        self.send = send
        self.clock = clock
        self.cooldown_s = cooldown_s
        self.system_name = system_name
        self.last_sent: dict[tuple[str, str], float] = {}
        self.waiting: dict[tuple[str, str], StockOut] = {}   # refusals inside the cooldown, told next time
        self.recent: collections.deque[dict] = collections.deque(maxlen=keep)
        self.messages = 0

    def saw(self, new_logs: pd.DataFrame) -> int:
        """Feeds newly read log lines; sends the messages that are due and returns how many were sent."""
        for item in stock_outs(new_logs):
            self.recent.append(item.view())
            key = (item.store_id, item.sku or item.product)
            held = self.waiting.get(key)
            if held is not None:
                item.refused += held.refused
            self.waiting[key] = item
        due: dict[str, list[tuple[tuple[str, str], StockOut]]] = {}
        for key, item in self.waiting.items():
            last = self.last_sent.get(key)
            if last is None or item.t - last >= self.cooldown_s:
                due.setdefault(item.store, []).append((key, item))
        sent = 0
        for store, entries in due.items():
            if self.send is not None:
                self.send(format_stock_out(store, [i for _, i in entries], self.clock, self.system_name))
            for key, item in entries:
                self.last_sent[key] = item.t
                del self.waiting[key]
            sent += 1
        self.messages += sent
        return sent

    def view(self, limit: int = 50) -> dict:
        return {"cooldown_min": self.cooldown_s / 60.0, "messages_sent": self.messages,
                "waiting": [i.view() for i in self.waiting.values()], "recent": list(self.recent)[-limit:][::-1]}
