"""Tests for morie.fn.kmsrv -- Kaplan-Meier survival curve."""

import numpy as np
import pytest

from morie.fn.kmsrv import kmsrv


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
    r = kmsrv(time, event)
    assert isinstance(r, dict)
    for k in ("times", "survival", "se", "ci_lower", "ci_upper", "n_obs", "n_events"):
        assert k in r


def test_survival_monotone(surv_data):
    time, event = surv_data
    r = kmsrv(time, event)
    assert np.all(np.diff(r["survival"]) <= 1e-12)


def test_survival_bounded(surv_data):
    time, event = surv_data
    r = kmsrv(time, event)
    assert np.all(r["survival"] >= 0)
    assert np.all(r["survival"] <= 1)


def test_se_nonneg(surv_data):
    time, event = surv_data
    r = kmsrv(time, event)
    assert np.all(r["se"] >= 0)


def test_empty_raises():
    with pytest.raises(ValueError):
        kmsrv(np.array([]), np.array([]))


def test_length_mismatch():
    with pytest.raises(ValueError):
        kmsrv(np.array([1, 2]), np.array([1]))


def test_cheatsheet():
    from morie.fn.kmsrv import cheatsheet
    assert "kaplan" in cheatsheet().lower()
