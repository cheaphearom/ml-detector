"""Reading the services' JSON log files and the experiment's fault labels.

Every service writes one JSON object per line (logstash-logback-encoder). Relevant fields:

    @timestamp, level, message, logger_name, service, traceId, spanId, stack_trace,
    event, method, path, status, durationMs, ...   (business fields from kv(...))

Each message is reduced to a *template* by masking UUIDs, business ids, timestamps and
numbers, so that "Saga started event=saga_started sagaId=<UUID> ..." is one template no matter
which saga it is about.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "ts", "t", "service", "level", "logger", "message", "template", "event",
    "method", "path", "status", "durationMs", "has_stack", "traceId",
]

_MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b[A-Z]{1,4}-\d+\b"), "<ID>"),
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "<HEX>"),
    (re.compile(r"(?<![A-Za-z<])-?\d+(?:\.\d+)?"), "<NUM>"),
]
_WS = re.compile(r"\s+")
MAX_TEMPLATE_LEN = 300


def make_template(message: str | None) -> str:
    """Masks variable parts of a log message so that equal messages share one template."""
    if not message:
        return ""
    text = str(message)
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    text = _WS.sub(" ", text).strip()
    return text[:MAX_TEMPLATE_LEN]


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def log_files(path: str | os.PathLike) -> list[Path]:
    """All log files under ``path`` (a file or a directory, searched recursively).

    Rolled-over files (``*.log.gz``) are included so long experiments are read completely.
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.exists():
        return []
    files = [f for f in p.rglob("*") if f.is_file()
             and (f.name.endswith(".log") or f.name.endswith(".log.gz") or f.name.endswith(".jsonl"))]
    return sorted(files)


def iter_records(lines: Iterable[str]) -> Iterator[dict]:
    """Parses JSON lines, silently skipping anything that is not a JSON object."""
    for line in lines:
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _to_float(value) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def records_to_frame(records: Iterable[dict], default_service: str | None = None) -> pd.DataFrame:
    """Converts raw log records into the normalised frame used by the feature extraction."""
    rows = []
    for r in records:
        ts = r.get("@timestamp") or r.get("timestamp")
        if not ts:
            continue
        message = r.get("message", "")
        rows.append((
            ts,
            r.get("service") or default_service or "unknown",
            str(r.get("level", "INFO")).upper(),
            r.get("logger_name", ""),
            message,
            r.get("event") or "",
            r.get("method") or "",
            r.get("path") or "",
            _to_float(r.get("status")),
            _to_float(r.get("durationMs")),
            bool(r.get("stack_trace")),
            r.get("traceId") or "",
        ))
    if not rows:
        return empty_logs()
    df = pd.DataFrame(rows, columns=["ts", "service", "level", "logger", "message", "event",
                                     "method", "path", "status", "durationMs", "has_stack", "traceId"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["ts"])
    df["t"] = to_epoch_seconds(df["ts"])
    df["template"] = [make_template(m) for m in df["message"]]
    return df[LOG_COLUMNS].sort_values("t", kind="stable").reset_index(drop=True)


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_epoch_seconds(ts: pd.Series) -> pd.Series:
    """Epoch seconds as float, independent of the datetime resolution pandas chose."""
    return (ts - _EPOCH) / pd.Timedelta(seconds=1)


def empty_logs() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ("t", "status", "durationMs"):
        df[c] = df[c].astype("float64")
    df["has_stack"] = df["has_stack"].astype("bool")
    return df


def load_logs(path: str | os.PathLike, start: pd.Timestamp | None = None,
              end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Loads every JSON log file under ``path`` into one time-sorted frame."""
    frames = []
    for f in log_files(path):
        service_hint = f.name.split(".")[0]
        with _open_text(f) as fh:
            frames.append(records_to_frame(iter_records(fh), default_service=service_hint))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_logs()
    df = pd.concat(frames, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    if start is not None:
        df = df[df["ts"] >= start]
    if end is not None:
        df = df[df["ts"] <= end]
    return df.reset_index(drop=True)


LABEL_COLUMNS = ["fault_id", "service", "type", "start", "end", "params"]


def load_labels(path: str | os.PathLike | None) -> pd.DataFrame:
    """Loads labels.csv written by the experiment runner. Missing file -> empty frame."""
    if path is None or not Path(path).exists():
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in LABEL_COLUMNS})
        df["start"] = pd.to_datetime(df["start"], utc=True)
        df["end"] = pd.to_datetime(df["end"], utc=True)
        return df
    df = pd.read_csv(path, dtype={"fault_id": str, "service": str, "type": str, "params": str},
                     keep_default_na=False)
    missing = {"fault_id", "service", "type", "start", "end"} - set(df.columns)
    if missing:
        raise ValueError(f"labels file {path} is missing columns: {sorted(missing)}")
    if "params" not in df.columns:
        df["params"] = ""
    df["start"] = pd.to_datetime(df["start"], utc=True, format="ISO8601")
    df["end"] = pd.to_datetime(df["end"], utc=True, format="ISO8601")
    return df[LABEL_COLUMNS].sort_values("start").reset_index(drop=True)


def load_experiment_meta(labels_path: str | os.PathLike | None) -> dict:
    """Reads experiment.json next to labels.csv, if present."""
    if labels_path is None:
        return {}
    meta_path = Path(labels_path).with_name("experiment.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class LogTailer:
    """Incrementally reads new lines from the log files in a directory (live detection).

    Keeps a byte offset per file, holds back an incomplete trailing line until it is finished,
    and starts over when a file shrinks (log rotation). Compressed rolled files are ignored:
    their content was already read while they were still the active file.
    """

    def __init__(self, path: str | os.PathLike, from_end: bool = False):
        self.path = Path(path)
        self.from_end = from_end
        self._offsets: dict[Path, int] = {}
        self._partial: dict[Path, bytes] = {}

    def _active_files(self) -> list[Path]:
        return [f for f in log_files(self.path) if not f.name.endswith(".gz")]

    def read_new(self) -> pd.DataFrame:
        frames = []
        for f in self._active_files():
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if f not in self._offsets:
                self._offsets[f] = size if self.from_end else 0
            offset = self._offsets[f]
            if size < offset:
                offset = 0
                self._partial.pop(f, None)
            if size == offset:
                continue
            with f.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
            self._offsets[f] = offset + len(chunk)
            data = self._partial.pop(f, b"") + chunk
            last_newline = data.rfind(b"\n")
            if last_newline < 0:
                self._partial[f] = data
                continue
            complete, rest = data[:last_newline + 1], data[last_newline + 1:]
            if rest:
                self._partial[f] = rest
            lines = complete.decode("utf-8", errors="replace").splitlines()
            frames.append(records_to_frame(iter_records(lines), default_service=f.name.split(".")[0]))
        frames = [fr for fr in frames if not fr.empty]
        if not frames:
            return empty_logs()
        return pd.concat(frames, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
