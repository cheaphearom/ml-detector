import datetime as dt
from pathlib import Path

import pytest

from detector.synthetic import generate


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory) -> Path:
    """A small synthetic experiment: ~45 min, faults in both the training and the test period."""
    out = tmp_path_factory.mktemp("synthetic")
    generate(out, duration_min=45, warmup_min=4, rps=4.0, seed=3, min_fault_s=60, max_fault_s=100,
             min_gap_s=120, max_gap_s=200, start=dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.timezone.utc))
    return out


@pytest.fixture(scope="session")
def second_synthetic_run(tmp_path_factory) -> Path:
    """Another small synthetic experiment (different seed and day), e.g. for cross-run tests."""
    out = tmp_path_factory.mktemp("synthetic2")
    generate(out, duration_min=45, warmup_min=4, rps=4.0, seed=7, min_fault_s=60, max_fault_s=100,
             min_gap_s=120, max_gap_s=200, start=dt.datetime(2026, 3, 4, 12, 0, tzinfo=dt.timezone.utc))
    return out


@pytest.fixture(autouse=True)
def _ecommerce_profile_by_default(monkeypatch):
    """Tests are written for the e-commerce profile unless they choose one, even where
    DETECTOR_PROFILE=pos is set (the POS Docker image)."""
    monkeypatch.setenv("DETECTOR_PROFILE", "ecommerce")
