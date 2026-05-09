"""Tests for moirais.fn.naest -- Nelson-Aalen cumulative hazard."""

import numpy as np
import pytest

from moirais.fn.naest import naest


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    t = rng.exponential(5, 80)
    c = rng.exponential(8, 80)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(surv_data):
    time, event = surv_data
    r = naest(time, event)
    assert isinstance(r, dict)
    for k in ("times", "cumulative_hazard", "se", "n_obs", "n_events"):
        assert k in r


def test_cumhaz_monotone(surv_data):
    time, event = surv_data
    r = naest(time, event)
    assert np.all(np.diff(r["cumulative_hazard"]) >= -1e-12)


def test_cumhaz_nonneg(surv_data):
    time, event = surv_data
    r = naest(time, event)
    assert np.all(r["cumulative_hazard"] >= 0)


def test_empty_raises():
    with pytest.raises(ValueError):
        naest(np.array([]), np.array([]))


def test_cheatsheet():
    from moirais.fn.naest import cheatsheet
    assert "nelson" in cheatsheet().lower()
