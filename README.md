# ML anomaly detector

Log-based anomaly detection with Telegram alerts. It reads the JSON logs of a system, scores every 10 seconds
whether the last 30 seconds look normal (fixed rules, Isolation Forest, PCA), explains what is wrong, and sends one
message when a problem starts and one when it is resolved.

It watches the **POS** ([pos-backend](https://github.com/cheaphearom/pos-backend)) and is also the detector of the
thesis project ([e-commerce](https://github.com/cheaphearom/e-commerce)); a *profile* says which system
(`DETECTOR_PROFILE=pos|ecommerce`). History: [ORIGIN.md](ORIGIN.md).

```
 POS server                                   this stack (same server, or another one)
 ──────────                                   ─────────────────────────────────────────
 pos-api ──► logs/pos-api.log
               │ Grafana Alloy ─────────────►  Loki ◄── ml-detector ──► Telegram
                                                 │        ▲  every 10 s: newest 30-s window
 pos-api /api/v1/monitoring/* ──────────────────┼────────┘  (rules, Isolation Forest, PCA)
   (System monitor page)                         ▼
                                              Grafana: search the logs, "POS logs" dashboard
```

| Part | |
|---|---|
| `detector/` | the detector (Python 3.12, FastAPI, scikit-learn, pandas) |
| `docker-compose.yml` | Loki 3.5, Grafana 12.1 and the detector; creates the Docker network `pos-monitoring` |
| `loki/loki.yaml` | Loki, single binary, logs kept `LOKI_RETENTION` (30 days) |
| `grafana/` | Loki data source and the "POS logs" and "ML detector" dashboards (provisioned) |
| `tests/` | pytest, with synthetic POS and e-commerce logs, a mock Telegram and a mock Loki |

## Run it

```bash
cp .env.example .env
```

Fill in at least `GRAFANA_ADMIN_PASSWORD` and `DETECTOR_API_KEY` (the same key as in the POS's `deploy/.env`), and
for alerts `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Then:

```bash
docker compose up -d --build
```

Start this stack **before** the POS's monitoring add-on: it creates the network `pos-monitoring`, which the POS
joins. On the POS server, in `deploy/.env` of pos-backend:

```
COMPOSE_FILE=docker-compose.yml:docker-compose.monitoring.yml
DETECTOR_API_KEY=<the same key>
```

and `docker compose up -d` there. Grafana Alloy then ships `deploy/logs` to Loki, and the API reaches the detector at
`http://ml-detector:8000`. Nothing is published to the internet: Grafana listens on `127.0.0.1:3001` only (an SSH
tunnel, or a Cloudflare Tunnel hostname protected by Cloudflare Access, to open it from elsewhere).

Memory: Loki ≈ 200–500 MB, Grafana ≈ 100–200 MB, the detector ≈ 200 MB plus ≈ 600 MB while training on a day.

**On another server:** run this stack there, give Loki and the detector a private way in (Tailscale / WireGuard, or
Cloudflare Tunnel + Access), and set on the POS server `LOKI_PUSH_URL` (Alloy) and `DETECTOR_URL` (the API).

**Development:** `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build` also publishes Loki
(`localhost:3100`) and the detector (`localhost:8000`, with its own small dashboard) for a POS API run with
`npm run dev`.

## How it decides

- **Features per window** (30 s, every 10 s): per service, log volume, ERROR / WARN lines, HTTP requests, 4xx,
  5xx and their share, latency (mean, p95, max), queue job p95, exceptions, distinct messages; system-wide, a count
  per business event of the profile, a success ratio and never-seen error messages.
- **POS profile**: services `pos-api` and `pos-worker`; events `sale_completed`, `sale_failed`, `payment_failed`,
  `stock_negative`, `cash_mismatch`, `webhook_failed`, `webhook_retries_exhausted`, `login_failed`,
  `db_query_slow`; `sale_success_ratio` = completed / (completed + failed + failed payments).
- **Stock-outs are business, not faults**: a sale refused with "Not enough stock for … (SKU-…)" becomes the event
  `sale_out_of_stock`. Its lines (the WARNs, the 400 response, the product name in the message) stay out of the
  service features, the success ratio and the never-seen messages; `stock_negative` still counts them. In the
  thesis runs this removed most of PCA's false alarms in run 3 (false-alarm windows 34 → 12, precision 0.66 →
  0.84) without missing a fault. The shop is told separately, see below.
- **Rules** (no training needed; each must hold for 3 windows): `pos-api` 5xx above 5 % (≥ 10 requests), p95 above
  1500 ms, more than 5 ERROR lines in a service, more than 3 failed payments, any webhook given up.
- **Isolation Forest and PCA** (reconstruction error and Hotelling's T² for the POS), trained on normal windows. The
  POS has no labelled faults, so Random Forest and Logistic Regression only run with a labels file (the thesis).
- **Notifications**: by default the rules and PCA (`NOTIFY_METHODS`); PCA must alert 4 windows in a row
  (`NOTIFY_MIN_WINDOWS`), so one stray error does not page anyone. In the POS a fault moves only a few features and
  Isolation Forest misses many of them: it is scored and shown, but does not notify.
- **Explanation**: the suspected service, the kind of problem (slow responses, errors, payment failures, webhook
  delivery failures, slow database, failed sales, many failed logins), the strongest signals against their normal
  level, real log lines, and the store when most failures come from one store.

## When the API is down, or its logs stop

The anomaly detection only sees what the logs say; a stopped API writes nothing. Two checks cover that, with the
same one-message-when-it-starts, one-when-it-ends rule as the anomaly alerts:

| Check | How | Message |
|---|---|---|
| **API probe** | `GET POS_HEALTH_URL` (`http://api:3000/health`) every 30 s; a failure is no answer, a timeout, or HTTP 503 (the POS's database or Redis down) | 🚨 "API not responding" after 2 failures in a row (≈ 1 min), with the reason (e.g. `HTTP 503 (database: fail)`); ✅ "API responding again" |
| **Heartbeat** | the POS API writes `event=heartbeat` every 30 s, even with no customers; the detector raises when none arrives for `HEARTBEAT_TIMEOUT_S` (120 s) | 🚨 "No logs from the API": the API is down, or Alloy / Loki do not deliver its logs (the reason says when the detector cannot read Loki); ✅ "Logs arriving again" |

Heartbeat lines are not part of the anomaly scores. The heartbeat check arms itself on the first heartbeat it sees,
so a system that does not write heartbeats never raises it. Both checks appear in **Recent alerts** on the POS's
System monitor page and in `/health` (`checks`).

When the **whole server** is down, the detector is down with it: add an outside uptime monitor (e.g. UptimeRobot,
free) on the public `https://api.yourcafe.com/health`, with its Telegram alerts.

## The public address, the server and the backups

Three more checks, same rule (one message when a problem starts, one when it ends), shown in **Recent alerts** on
the POS's System monitor page and in `/health` (`checks`):

| Check | How | Message |
|---|---|---|
| **Public address** | `GET PUBLIC_HEALTH_URL` every 60 s: the POS's `/health` through the address the tills use (Tailscale Funnel, Cloudflare Tunnel). Empty = off | 🚨 "POS not reachable from outside" after 2 failures. Without an "API not responding" message the API is fine: the tunnel or the web app is the problem |
| **Disk** | every 60 s, the disk holding `./models` (the server's disk) | 🚨 "Server disk almost full" above `DISK_ALERT_PCT` (85 %) twice in a row, with the command that frees space |
| **Memory** | every 60 s, `/proc/meminfo` (a container sees the server's memory) | 🚨 "Server memory almost full" when less than `MEMORY_ALERT_PCT` (10 %) is available for 5 minutes |
| **Backups** | every 10 min, the newest `pos-*.sql.gz` in `../pos-backend/deploy/backups` (mounted read only; `POS_BACKUP_DIR` for another place) | 🚨 "Database backup problem" when it is older than `BACKUP_MAX_AGE_H` (26 h), almost empty (a failed dump) or less than half the size of the one before |

The backup check arms itself on the first backup file it sees, so a wrong folder never raises it: `/health` then
says "waiting for a first backup". Containers are not watched one by one (that would need the Docker socket, which
gives full control of the server): the API probe covers the API, database and Redis; the public address the web
app and the tunnel; the heartbeat the log shipping.

## Out of stock (business alert)

When a cashier cannot sell a product because the store has none left, Telegram gets one message per store:

```
🛒 Out of stock: Phnom Penh - Toul Kork (POS)
Time: 2026-11-27 17:24:32 Asia/Phnom_Penh

• Dish Soap 500ml (SKU-0022): 0 left, a customer wanted 2, 1 sale refused
```

The same product in the same store is reported again after `STOCK_ALERT_COOLDOWN_MIN` (60), with the number of
sales refused in between. `GET /stock-outs` lists the recent ones. `STOCK_ALERTS=off` turns the messages off.

## The detector's own dashboard

The detector writes its own activity to Loki (`job="ml-detector"`, never mixed with the POS logs): one line per
10-second poll (lines read, lag, every method's score and threshold, heartbeat age, POS API response time, Telegram
queue), plus alerts, incidents, check changes and training. Grafana's **ML detector** dashboard (folder POS) shows
polls per minute, lines read per minute, heartbeat age, POS API health-check time, score ÷ threshold per method,
alerts per method, incidents and checks, detection lag, Telegram, and an event list. Writing is best effort: if Loki
does not answer, these lines are dropped and the detection goes on. `SELF_MONITORING=off` turns it off.

## Training

Until it is trained only the rules run. **Automatic training** (on in `docker-compose.yml`) takes care of it:

- **the first model** as soon as there is a day of logs (`MIN_TRAIN_HOURS`, 24), at `AUTO_TRAIN_AT` (04:30, in
  `NOTIFY_TIMEZONE`, a quiet hour);
- **then a new one every week** (`AUTO_TRAIN_EVERY_DAYS`), on the last `AUTO_TRAIN_HOURS` (24) hours, so the model
  follows new products, stores and seasons;
- it runs **in the background**: detection goes on with the old model and switches when the new one is ready;
- **safety**: windows where the rules fire, and every period the detector alerted on (kept in
  `models/incidents.json`, also across restarts), are left out; while an incident is open it waits for the next day;
  if more than `AUTO_TRAIN_MAX_UNUSUAL` (20 %) of the period looked unusual, **the old model stays**;
- a Telegram message says when it trained ("🧠 Detector re-trained…") or why it skipped ("⚠️ … training
  skipped"); `/health` → `auto_training` shows the last result and the next try.

By hand (e.g. right after a big change), **after at least 1 day of normal traffic**, on a period without incidents
(windows where the rules fire are left out automatically):

```bash
docker compose exec ml-detector python -m detector.cli train --hours 24
```

With `LOG_SOURCE=loki` it reads the period from Loki; the running detector loads the new models within 10 seconds
(kept in `./models`). Train on 24–48 hours. Train again after big changes: a new store, new opening hours, a much
busier season.

## Telegram

1. In Telegram, open **@BotFather**, send `/newbot` and follow the steps. Copy the **token**.
2. Open your new bot and send it any message (e.g. `hi`). For a group: add the bot and send a message there.
3. Find the **chat id**: open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy the number at
   `"chat":{"id": ...}` (group ids start with `-`).
4. Put `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` and recreate the detector:

```bash
docker compose up -d ml-detector
```

5. Test it with the **Send test Telegram message** button on the POS's System monitor page.

If Telegram cannot be reached the detector logs a warning, keeps running and sends the message later (10 s, 30 s,
1, 2, then every 5 minutes, for up to 6 hours). Times in the messages use `NOTIFY_TIMEZONE`.

## Settings (`.env`)

| Variable | Default | |
|---|---|---|
| `GRAFANA_ADMIN_PASSWORD` | – (required) | Grafana user `admin` |
| `GRAFANA_PORT`, `GRAFANA_ROOT_URL` | 3001, `http://localhost:3001` | |
| `LOKI_RETENTION` | `720h` | how long Loki keeps logs |
| `DETECTOR_PROFILE` | `pos` | `ecommerce`: the thesis system |
| `DETECTOR_API_KEY` | empty | header `X-API-Key` on every endpoint but `/health`; recommended |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | empty = no Telegram | |
| `NOTIFY_METHODS` | profile (`rules,pca`) | detectors that may send alerts |
| `NOTIFY_RESOLVE_AFTER_S` | 60 | seconds without an alert before "resolved" |
| `NOTIFY_MIN_WINDOWS` | profile (4) | windows in a row an ML detector must alert before a message |
| `NOTIFY_TIMEZONE` | UTC | |
| `STOCK_ALERTS`, `STOCK_ALERT_COOLDOWN_MIN` | `on`, 60 | "Out of stock" messages; minutes before the same product in a store is reported again |
| `LOKI_QUERY` | `{job="pos"}` | LogQL stream selector of the system's logs |
| `LOKI_DELAY_S` | 10 | read up to this many seconds ago (lines reach Loki a little later) |
| `POS_HEALTH_URL` | `http://api:3000/health` | the API probe; empty = off |
| `HEALTH_CHECK_INTERVAL_S`, `HEALTH_FAIL_AFTER` | 30, 2 | probe every 30 s; "not responding" after 2 failures |
| `HEARTBEAT_TIMEOUT_S` | 120 | "no logs" after this many seconds without a heartbeat; 0 = off |
| `SELF_MONITORING` | `on` | the detector's own lines to Loki (ML detector dashboard) |
| `AUTO_TRAIN`, `AUTO_TRAIN_EVERY_DAYS`, `AUTO_TRAIN_AT`, `AUTO_TRAIN_HOURS`, `AUTO_TRAIN_MAX_UNUSUAL` | `on` (compose), 7, `04:30`, 24, 0.2 | automatic training |
| `PUBLIC_HEALTH_URL`, `PUBLIC_CHECK_INTERVAL_S` | empty, 60 | the POS through its public address; empty = off |
| `HOST_CHECKS`, `DISK_ALERT_PCT`, `MEMORY_ALERT_PCT` | `on`, 85, 10 | the server's disk and memory |
| `POS_BACKUP_DIR`, `BACKUP_MAX_AGE_H` | `../pos-backend/deploy/backups`, 26 | the POS database backups |

Without Docker or without Loki, the detector also reads log files: `LOG_SOURCE=files` (the default outside
`docker-compose.yml`) and `LOG_DIR`. Other variables: `LOKI_URL`, `LOKI_ORG_ID`, `LOKI_USERNAME`, `LOKI_PASSWORD`,
`MODEL_DIR`, `WINDOW`, `STEP`, `LIVE_INTERVAL`, `BUFFER_MINUTES`, `MIN_TRAIN_HOURS`, `TELEGRAM_API_URL`.

## API

| Endpoint | |
|---|---|
| `GET /health` | status, profile, log source, models, last window, Telegram queue (no key needed) |
| `GET /detect` | the newest window: every method's score, threshold and alert, key features, explanation |
| `GET /alerts`, `GET /scores?minutes=30` | recent alerts; scores per window |
| `GET /stock-outs` | recent sales refused for lack of stock, per store and product |
| `POST /train` | `{"hours": 24}` or `{"start": "…", "end": "…"}` |
| `POST /notify/test` | a test message to Telegram |
| `POST /evaluate` | offline evaluation of labelled experiments (the thesis) |

The POS's System monitor page shows all of this through the POS API; the browser never calls the detector.

## Thesis evaluation (e-commerce profile)

```bash
python -m detector.cli evaluate --logs ../results/run_1/logs --labels ../results/run_1/labels.csv --out ../results/run_1/detector
```

`synth` (synthetic runs), `aggregate` (several runs) and `cross` (train on one run, test on another) are described
in the e-commerce repository's README.

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest
```

The tests use synthetic logs, a mock Telegram Bot API and a mock Loki. With a real Loki:
`LOKI_TEST_URL=http://localhost:3100 pytest tests/test_loki.py`.
