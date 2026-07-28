"""Tests for the calibration-period slice used by the HBV loss.

Covers the calibration-period -> post-warmup index slicing used by the
gradient/value-and-grad loss (jhbv.calibration.worker._build_loss_fn). The
slice itself lives on the shared InMemoryModelWorker base; HBV supplies the
timestep-aware warmup length via warmup_steps().
"""

import numpy as np
import pandas as pd
import pytest

from jhbv.calibration.worker import HBVWorker

# get_calibration_slice() landed in SYMFLUENCE alongside this change
# (symfluence-org/SYMFLUENCE#392). Until a release carrying it is on PyPI,
# CI resolves an older symfluence and these cases cannot run. They activate
# on their own once the dependency catches up — no follow-up edit needed.
pytestmark = pytest.mark.skipif(
    not hasattr(HBVWorker, "get_calibration_slice"),
    reason="requires a symfluence providing InMemoryModelWorker.get_calibration_slice",
)


def _make_worker(time_index, cal_period, warmup_days=365, timestep_hours=24):
    """Build a bare HBVWorker with only the attributes the method reads."""
    w = HBVWorker.__new__(HBVWorker)
    w._time_index = time_index
    w.warmup_days = warmup_days
    w.timestep_hours = timestep_hours
    # Stub the config accessor: return the calibration period, else the default.
    cfg = {"CALIBRATION_PERIOD": cal_period} if cal_period is not None else {}
    w._cfg = lambda key, default=None: cfg.get(key, default)
    return w


def test_slice_selects_the_calibration_window_after_warmup():
    # 365 warmup days (2010) + a full 2011 year of daily steps.
    idx = pd.date_range("2010-01-01", periods=365 + 365, freq="D")
    w = _make_worker(idx, "2011-03-01,2011-05-31")

    sl = w.get_calibration_slice()
    assert sl is not None
    start, end = sl

    # Indices are relative to the POST-warmup array (2011-01-01 == index 0).
    after_warmup = idx[365:]
    assert after_warmup[start] == pd.Timestamp("2011-03-01")
    assert after_warmup[end - 1] == pd.Timestamp("2011-05-31")
    # Mar 1 -> May 31 inclusive = 92 days.
    assert end - start == 92


def test_no_calibration_period_returns_none():
    idx = pd.date_range("2011-01-01", periods=400, freq="D")
    assert _make_worker(idx, None).get_calibration_slice() is None
    assert _make_worker(idx, "").get_calibration_slice() is None


def test_missing_time_index_returns_none():
    assert _make_worker(None, "2011-03-01,2011-05-31").get_calibration_slice() is None


def test_period_outside_record_returns_none():
    idx = pd.date_range("2010-01-01", periods=365 + 365, freq="D")
    # A window entirely before the post-warmup span yields no matching indices.
    assert _make_worker(idx, "2009-01-01,2009-06-30").get_calibration_slice() is None


def test_malformed_period_returns_none():
    idx = pd.date_range("2010-01-01", periods=365 + 365, freq="D")
    assert _make_worker(idx, "not-a-date").get_calibration_slice() is None
    assert _make_worker(idx, "2011-01-01").get_calibration_slice() is None  # single date
