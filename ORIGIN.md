# Origin

Copied from the `ml-detector/` folder of the thesis project
[cheaphearom/e-commerce](https://github.com/cheaphearom/e-commerce) at commit `7ec9f0c` (2026-11-12, `main`).
The full history stays in that repository; this copy continues it for the POS.

## History in the e-commerce repository

| Commit | Date | Change |
|---|---|---|
| `35bf7dd` | 2026-11-08 | Saga microservices, fault injection and the log anomaly detector |
| `0b2eada`, `d4c8bb2` | 2026-11-08 | Docker builds that survive slow connections |
| `7bb7f6b` | 2026-11-08 | Explanations of alerts, Telegram notifications |
| `344be4e` | 2026-11-08 | No overlapping experiments, no evaluating unfinished runs |
| `0789cde` | 2026-11-08 | Monitoring dashboard |
| `e9f3bcf` | 2026-11-10 | Rule baseline tuned on training windows (three rule variants) |
| `b969be1` | 2026-11-10 | PCA and Logistic Regression detectors, grid-edge check |
| `4283cb3`, `8fbdab4` | 2026-11-10 | Aggregation over runs, bootstrap confidence intervals, paired tests |
| `729fa69` | 2026-11-10 | Cross-run evaluation (train on one run, test on another) |
| `2776b45` | 2026-11-10 | Script for all five experiments |
| `4174715` | 2026-11-11 | CORS for the Flutter app |

`tests/test_run_all_script.py` tests `scripts/run_all_experiments.sh` of the e-commerce repository; the script was
not copied, so those 7 tests are skipped here.

## Changes for the POS (this repository)

- **Profiles** (`detector/profiles.py`, `DETECTOR_PROFILE=ecommerce|pos`, default `ecommerce`): services, the entry
  service checked by the rules, the business events counted as system features, the success ratio, the symptom
  events, rule thresholds and extra event rules. Every hard-coded service name and saga feature now comes from the
  profile; the e-commerce profile gives exactly the old behaviour.
- **POS profile**: services `pos-api` and `pos-worker`; `sale_success_ratio` =
  `sale_completed / (sale_completed + sale_failed + payment_failed)`; rules: 5xx > 5 % (≥ 10 requests), p95 > 1500 ms,
  > 5 ERROR lines, > 3 `payment_failed`, any `webhook_retries_exhausted`, each for 3 windows in a row.
- **PCA with Hotelling's T²** (POS profile only): in the POS a fault often moves just one group of features (e.g.
  only `pos-api` latency), which the reconstruction error alone does not see.
- `--profile` option of the CLI; the dashboard's service health and fault injection are e-commerce only.
- `tests/test_pos_profile.py` and `tests/pos_logs.py` (synthetic POS logs).
- **Live detection for the POS**: PCA runs in the live loop; per-profile `live_methods` and `notify_methods`
  (POS: alerts from the rules and PCA only; Isolation Forest is scored and shown but does not notify).
  Random Forest only trains when a labels file with faults exists.
- **Training without labels**: windows where the rules fire are left out, so a known incident is not learned as
  normal; `{"hours": N}` on `POST /train` and `python -m detector.cli train --hours N` (the running detector
  reloads the model file); a warning when less than `MIN_TRAIN_HOURS` (24) of logs were used.
- `GET /scores`, `last_window_end` and `notify_methods` in `/health`, optional `DETECTOR_API_KEY`
  (header `X-API-Key` on every endpoint except `/health`).
- **POS alert wording**: profile `event_problems` name the problem from business events (webhook delivery
  failures, slow database, payment failures, failed sales, many failed logins) and the service when none stands
  out; `kind_texts` replace the thesis fault names (LATENCY, ERROR, DB_TIMEOUT) in POS messages. The explanation
  names the store when most WARN/ERROR lines of the window come from one store (parsing keeps `storeId`,
  `storeName`). Messages say "in the POS"; `NOTIFY_TIMEZONE` sets the time zone of their times.
- **Telegram retries**: messages are queued; while Telegram is unreachable (network, 5xx, 429) they are sent
  again after 10 s, 30 s, 1, 2, then every 5 min (at most 6 h), with a WARN log line each time; refused messages
  (wrong token / chat) are dropped with a WARN. `/health` shows pending / sent / dropped / last error.
  `TELEGRAM_API_URL` (tests use a local mock of the Bot API).

