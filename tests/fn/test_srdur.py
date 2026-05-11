"""Tests for morie.fn.srdur -- Survival duration statistics."""

import numpy as np
import pytest

from morie.fn.srdur import srdur


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 100
    t = rng.exponential(5, n)
    c = rng.exponential(8, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(surv_data):
    time, event = surv_data
    r = srdur(time, event)
    assert isinstance(r, dict)
    for k in ("median_survival", "percentile_times", "mean_restricted"):
        assert k in r


def test_percentiles_dict(surv_data):
    time, event = surv_data
    r = srdur(time, event)
    assert 25 in r["percentile_times"]
    assert 50 in r["percentile_times"]
    assert 75 in r["percentile_times"]


def test_mean_restricted_positive(surv_data):
    time, event = surv_data
    r = srdur(time, event)
    assert r["mean_restricted"] > 0


def test_cheatsheet():
    from morie.fn.srdur import cheatsheet
    assert "duration" in cheatsheet().lower()
