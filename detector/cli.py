"""Command line interface.

    python -m detector.cli --profile pos evaluate --logs ...   (--profile defaults to DETECTOR_PROFILE / ecommerce)
    python -m detector.cli --profile pos train --hours 24      (live detector: train on the last 24 h of logs)
    python -m detector.cli evaluate --logs ../logs --labels ../results/labels.csv --out ../results/eval
    python -m detector.cli synth --out ./synthetic --duration-min 120 --seed 1
    python -m detector.cli aggregate --runs ../results/run_* --out ../results/summary
    python -m detector.cli cross --train-run ../results/run_1 --test-run ../results/run_2 --out ../results/cross_1_2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .aggregate import aggregate
from .evaluate import cross_paths, evaluate_paths
from .models import METHOD_LABELS, METHOD_ORDER
from .profiles import PROFILES
from .synthetic import generate


def _summary(results: dict) -> str:
    lines = [f"{'method':<26}{'P':>7}{'R':>7}{'F1':>7}{'AUC':>7}{'FPR':>7}{'det.':>8}{'mean delay':>12}"]
    for m in [m for m in METHOD_ORDER if m in results["metrics"]]:
        r, d = results["metrics"][m], results["detection_delay"][m]
        auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "-"
        delay = f"{d['mean_delay_s']:.1f}s" if d["mean_delay_s"] is not None else "-"
        lines.append(f"{METHOD_LABELS[m]:<26}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}{auc:>7}"
                     f"{r['false_positive_rate']:>7.3f}{d['detected']:>4}/{d['faults']:<3}{delay:>12}")
    variants = results.get("comparison_vs_rule_variants") or {"rules": results["comparison_vs_rules"]}
    for rule, per_ml in variants.items():
        for m, c in per_ml.items():
            saved = f"{c['mean_time_saved_s']:.1f}s" if c["mean_time_saved_s"] is not None else "-"
            lines.append(f"{METHOD_LABELS[m]} vs {rule}: earlier={c['earlier']} same={c['same']} "
                         f"later_or_missed={c['later_or_missed']} both_missed={c['both_missed']} mean_saved={saved}")
    tuning = results.get("rule_tuning")
    if tuning:
        sel = ", ".join(f"{k}={v}" for k, v in tuning["selected"].items())
        lines.append(f"tuned rules (training windows only): {sel} "
                     f"(train F1 {tuning['train_f1']:.3f}, {tuning['train_false_alarm_episodes']} false-alarm episodes)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="detector", description="Log anomaly detection (e-commerce saga system or POS)")
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        help="system the logs come from (default: DETECTOR_PROFILE, else ecommerce)")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="compare the rule variants and the ML models")
    ev.add_argument("--logs", required=True, help="log directory (or a single log file)")
    ev.add_argument("--labels", required=True, help="labels.csv written by the experiment runner")
    ev.add_argument("--out", required=True, help="output folder for results.json, report.md and figures")
    ev.add_argument("--window", type=float, default=30.0, help="window length in seconds")
    ev.add_argument("--step", type=float, default=10.0, help="window step in seconds")
    ev.add_argument("--train-frac", type=float, default=0.6, help="chronological share used for training")
    ev.add_argument("--grace", type=float, default=15.0, help="seconds after a fault still labelled anomalous")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--start", help="ISO timestamp; default: load_start from experiment.json")
    ev.add_argument("--end", help="ISO timestamp; default: load_end from experiment.json")
    ev.add_argument("--no-figures", action="store_true")
    ev.add_argument("--allow-incomplete", action="store_true",
                    help="evaluate an experiment that is still running or was interrupted")
    ev.add_argument("--json", action="store_true", help="print results.json instead of the summary")
    ev.add_argument("--no-rule-tuning", action="store_true",
                    help="only the default rules (skip rules_tuned and rules_tuned_k1)")
    ev.add_argument("--methods", help=f"comma-separated subset of: {', '.join(METHOD_ORDER)} "
                                      f"(default: all; the default rules are always included)")

    ag = sub.add_parser("aggregate", help="summarise several evaluated runs (mean ± SD, pooled delays)")
    ag.add_argument("--runs", nargs="+", required=True,
                    help="run folders or glob patterns, each with detector/results.json and detection_delays.csv")
    ag.add_argument("--out", required=True, help="output folder for the summary tables and figures")
    ag.add_argument("--detector-dir", default="detector", help="evaluation sub-folder inside each run")
    ag.add_argument("--no-figures", action="store_true")
    ag.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for the confidence intervals")
    ag.add_argument("--seed", type=int, default=42, help="seed of the bootstrap")

    tr = sub.add_parser("train", help="train the live detector's models on the logs (the running API reloads them)")
    tr.add_argument("--logs", default=os.environ.get("LOG_DIR", "/logs"), help="log directory (default: LOG_DIR)")
    tr.add_argument("--models", default=os.environ.get("MODEL_DIR", "/models"), help="model folder (default: MODEL_DIR)")
    tr.add_argument("--hours", type=float, help="only the last N hours of logs (default: all)")
    tr.add_argument("--start", help="ISO timestamp of the first log line to use")
    tr.add_argument("--end", help="ISO timestamp of the last log line to use")
    tr.add_argument("--labels", help="labels.csv with faults (enables Random Forest); the POS has none")

    cr = sub.add_parser("cross", help="train every method on one whole run and test it on another whole run")
    cr.add_argument("--train-run", required=True, help="run folder with logs/, labels.csv and experiment.json")
    cr.add_argument("--test-run", required=True, help="another run folder with the same layout")
    cr.add_argument("--out", required=True, help="output folder for results.json, report.md and figures")
    cr.add_argument("--window", type=float, default=30.0)
    cr.add_argument("--step", type=float, default=10.0)
    cr.add_argument("--grace", type=float, default=15.0)
    cr.add_argument("--seed", type=int, default=42)
    cr.add_argument("--logs-subdir", default="logs", help="log folder inside each run folder")
    cr.add_argument("--methods", help="comma-separated subset of methods (default: all)")
    cr.add_argument("--no-rule-tuning", action="store_true")
    cr.add_argument("--no-figures", action="store_true")
    cr.add_argument("--allow-incomplete", action="store_true")
    cr.add_argument("--json", action="store_true", help="print results.json instead of the summary")

    sy = sub.add_parser("synth", help="generate synthetic logs (testing only, not for thesis results)")
    sy.add_argument("--out", required=True)
    sy.add_argument("--duration-min", type=float, default=120.0)
    sy.add_argument("--warmup-min", type=float, default=10.0)
    sy.add_argument("--rps", type=float, default=5.0)
    sy.add_argument("--seed", type=int, default=1)
    sy.add_argument("--min-fault-s", type=int, default=60)
    sy.add_argument("--max-fault-s", type=int, default=180)
    sy.add_argument("--min-gap-s", type=int, default=180)
    sy.add_argument("--max-gap-s", type=int, default=420)
    sy.add_argument("--ramp-frac", type=float, default=0.4, help="share of a fault spent ramping up")

    args = parser.parse_args(argv)
    if args.profile:
        os.environ["DETECTOR_PROFILE"] = args.profile
    if args.command == "train":
        from .api import DetectorState, Settings

        settings = Settings(log_dir=args.logs, model_dir=args.models, live=False,
                            labels_path=args.labels or os.environ.get("LABELS_PATH", "/nonexistent/labels.csv"))
        try:
            result = DetectorState(settings, send=lambda _text: None).train(
                args.logs, settings.labels_path, args.start, args.end, args.hours)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str))
        if result.get("warning"):
            print(f"warning: {result['warning']}", file=sys.stderr)
        print(f"models saved to {args.models}; the running detector loads them within one poll", file=sys.stderr)
        return 0
    if args.command == "evaluate":
        try:
            results = evaluate_paths(args.logs, args.labels, out_dir=args.out, window=args.window, step=args.step,
                                     train_frac=args.train_frac, grace=args.grace, seed=args.seed,
                                     start=args.start, end=args.end, make_figures=not args.no_figures,
                                     allow_incomplete=args.allow_incomplete,
                                     tune_rules=not args.no_rule_tuning,
                                     methods=[m.strip() for m in args.methods.split(",")] if args.methods else None)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(results, indent=2, default=str) if args.json else _summary(results))
        print(f"\nwritten to {args.out}", file=sys.stderr)
        return 0
    if args.command == "cross":
        try:
            results = cross_paths(args.train_run, args.test_run, out_dir=args.out, window=args.window,
                                  step=args.step, grace=args.grace, seed=args.seed,
                                  make_figures=not args.no_figures, allow_incomplete=args.allow_incomplete,
                                  tune_rules=not args.no_rule_tuning, logs_subdir=args.logs_subdir,
                                  methods=[m.strip() for m in args.methods.split(",")] if args.methods else None)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(results, indent=2, default=str) if args.json else _summary(results))
        print(f"\nwritten to {args.out}", file=sys.stderr)
        return 0
    if args.command == "aggregate":
        try:
            result = aggregate(args.runs, args.out, make_figures=not args.no_figures,
                               detector_dir=args.detector_dir, n_boot=args.bootstrap, seed=args.seed)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print((Path(args.out) / "summary_metrics.md").read_text(encoding="utf-8"))
        hypotheses = (Path(args.out) / "stats.md").read_text(encoding="utf-8").split("## Hypotheses")[1]
        print("Hypotheses" + hypotheses.split("\n## ")[0])
        print(f"{len(result['runs'])} runs aggregated, written to {args.out}", file=sys.stderr)
        return 0
    if args.command == "synth":
        meta = generate(args.out, duration_min=args.duration_min, warmup_min=args.warmup_min, rps=args.rps,
                        seed=args.seed, min_fault_s=args.min_fault_s, max_fault_s=args.max_fault_s,
                        min_gap_s=args.min_gap_s, max_gap_s=args.max_gap_s, ramp_frac=args.ramp_frac)
        print(f"{meta['faults']} faults over {meta['duration_min']} min written to {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
