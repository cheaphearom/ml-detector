"""Stock-outs: a business alert to the shop, not a technical symptom for the anomaly detection."""
import json

import numpy as np
from fastapi.testclient import TestClient

from detector.api import Settings, create_app
from detector.business import StockOutWatch, stock_outs
from detector.features import add_unseen_templates, build_windows, known_templates, system_name
from detector.parsing import STOCK_OUT_EVENT, iter_records, records_to_frame
from detector.profiles import get_profile

from pos_logs import pos_lines

POS = get_profile("pos")
STORE = "d165b4fa-e76f-4118-a2df-50d1e57697ee"


def stock_out_lines(ts: str, trace: str, product="Dish Soap 500ml", sku="SKU-0022", available=0, needed=2,
                    store=STORE, store_name="Phnom Penh - Toul Kork") -> list[str]:
    """The three lines the POS API writes for one sale refused for lack of stock (as in thesis run 3)."""
    reason = f"Not enough stock for {product} ({sku}): available {available}, needed {needed}"
    base = {"@timestamp": ts, "service": "pos-api", "traceId": trace}
    return [json.dumps(r) for r in (
        {**base, "level": "WARN", "logger_name": "BusinessEvent", "event": "stock_negative", "storeId": store,
         "productId": "p-1", "movement": "SALE", "quantity": -needed, "available": available, "blocked": True,
         "storeName": store_name, "message": f"Not enough stock event=stock_negative storeId={store} blocked=true"},
        {**base, "level": "WARN", "logger_name": "BusinessEvent", "event": "sale_failed", "storeId": store,
         "status": 400, "exception": "BadRequestException", "reason": reason, "storeName": store_name,
         "message": f"Sale failed event=sale_failed storeId={store} status=400 reason={reason}"},
        {**base, "level": "WARN", "logger_name": "HttpAccess", "event": "http_request", "method": "POST",
         "path": "/api/v1/sales", "status": 400, "durationMs": 31,
         "message": "HTTP request event=http_request method=POST path=/api/v1/sales status=400 durationMs=31"},
    )]


def frame(lines):
    return records_to_frame(iter_records(lines))


def test_refused_sale_becomes_a_business_stock_out():
    logs = frame(stock_out_lines("2026-11-27T10:24:32.797Z", "t1"))
    assert list(logs["event"]) == ["stock_negative", STOCK_OUT_EVENT, "http_request"]
    assert logs["business"].all()          # the 400 response shares the trace, so it is business too
    other = frame([json.dumps({"@timestamp": "2026-11-27T10:00:00Z", "service": "pos-api", "level": "WARN",
                               "event": "sale_failed", "reason": "Card declined", "message": "Sale failed"})])
    assert other["event"].iloc[0] == "sale_failed" and not other["business"].iloc[0]


def test_stock_outs_stay_out_of_the_technical_features():
    normal = pos_lines(300, seed=5)
    shortage = normal + stock_out_lines("2026-11-12T06:02:30.000Z", "t1") \
        + stock_out_lines("2026-11-12T06:02:35.000Z", "t2", product="Coca-Cola 330ml", sku="SKU-0101")
    shortage.sort(key=lambda line: json.loads(line)["@timestamp"])
    a, b = frame(normal), frame(shortage)
    start, end = float(a["t"].min()), float(a["t"].max())
    wa = build_windows(a, profile=POS, start=start, end=end)
    wb = build_windows(b, profile=POS, start=start, end=end)
    technical = [c for c in wa.columns if c.startswith("pos-api__")] + [system_name("sale_failed"),
                                                                       system_name("sale_success_ratio")]
    for c in technical:
        assert np.allclose(wa[c], wb[c]), c
    assert wb[system_name("stock_negative")].sum() > wa[system_name("stock_negative")].sum()
    # new product names in the messages are not "unseen templates" either
    known = known_templates(a)
    assert add_unseen_templates(wb, b, known)[system_name("unseen_templates")].sum() == \
        add_unseen_templates(wa, a, known)[system_name("unseen_templates")].sum()


def test_stock_outs_are_grouped_per_store_and_product():
    logs = frame(stock_out_lines("2026-11-27T10:24:32Z", "t1")
                 + stock_out_lines("2026-11-27T10:25:00Z", "t2", needed=1)
                 + stock_out_lines("2026-11-27T10:26:00Z", "t3", product="Rice 5kg", sku="SKU-0007", available=1))
    found = {i.sku: i for i in stock_outs(logs)}
    assert set(found) == {"SKU-0022", "SKU-0007"}
    assert found["SKU-0022"].refused == 2 and found["SKU-0022"].needed == 1   # the newest refusal
    assert found["SKU-0007"].available == 1 and found["SKU-0007"].store == "Phnom Penh - Toul Kork"


def test_watch_sends_one_message_per_store_then_waits_for_the_cooldown():
    sent = []
    watch = StockOutWatch(sent.append, clock=lambda e: f"@{e:.0f}", cooldown_s=3600)
    first = frame(stock_out_lines("2026-11-27T10:24:32Z", "t1")
                  + stock_out_lines("2026-11-27T10:24:40Z", "t2", product="Rice 5kg", sku="SKU-0007"))
    assert watch.saw(first) == 1
    msg = sent[0]
    assert msg.startswith("🛒 Out of stock: Phnom Penh - Toul Kork")
    assert "• Dish Soap 500ml (SKU-0022): 0 left, a customer wanted 2, 1 sale refused" in msg
    assert "Rice 5kg (SKU-0007)" in msg and "not a system fault" in msg

    # ten minutes later the same product again: no new message, but the refusal is remembered
    assert watch.saw(frame(stock_out_lines("2026-11-27T10:34:32Z", "t3"))) == 0
    # another store is told at once
    assert watch.saw(frame(stock_out_lines("2026-11-27T10:35:00Z", "t4", store="s2", store_name="Siem Reap"))) == 1
    assert sent[-1].startswith("🛒 Out of stock: Siem Reap")
    # after the cooldown: one message with both refusals counted
    assert watch.saw(frame(stock_out_lines("2026-11-27T11:30:00Z", "t5"))) == 1
    assert "Dish Soap 500ml (SKU-0022): 0 left, a customer wanted 2, 2 sales refused" in sent[-1]
    assert len(sent) == 3 and watch.view()["messages_sent"] == 3


def test_live_loop_sends_stock_out_alerts(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    lines = pos_lines(120, seed=3) + stock_out_lines("2026-11-12T06:01:00.000Z", "t1")
    lines.sort(key=lambda line: json.loads(line)["@timestamp"])
    (logs / "pos-api.log").write_text("\n".join(lines) + "\n")
    sent = []
    settings = Settings(profile="pos", log_dir=str(logs), model_dir=str(tmp_path / "m"), live=False,
                        tail_from_start=True, labels_path=str(tmp_path / "none.csv"), heartbeat_timeout_s=0)
    with TestClient(create_app(settings, send=sent.append)) as c:
        c.get("/detect")
        stock = [m for m in sent if m.startswith("🛒")]
        assert len(stock) == 1 and "Dish Soap 500ml (SKU-0022)" in stock[0]
        body = c.get("/stock-outs").json()
        assert body["enabled"] and body["recent"][0]["product"] == "Dish Soap 500ml"
        assert not any(m.startswith("🚨") for m in sent)


def test_stock_alerts_can_be_switched_off(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pos-api.log").write_text("\n".join(stock_out_lines("2026-11-12T06:01:00.000Z", "t1")) + "\n")
    sent = []
    settings = Settings(profile="pos", log_dir=str(logs), model_dir=str(tmp_path / "m"), live=False,
                        tail_from_start=True, labels_path=str(tmp_path / "none.csv"), stock_alerts=False)
    with TestClient(create_app(settings, send=sent.append)) as c:
        c.get("/detect")
        assert not sent and c.get("/stock-outs").json()["recent"]
