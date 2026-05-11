"""Tests for morie.fn.chzrd -- Cause-specific hazard ratio."""

import numpy as np
import pytest

from morie.fn.chzrd import chzrd


@pytest.fixture()
def cr_data():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    event = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    time = rng.exponential(5, n)
    return time, event, X


def test_returns_dict(cr_data):
    time, event, X = cr_data
    r = chzrd(time, event, X, cause=1)
    assert isinstance(r, dict)
    for k in ("coefficients", "se", "hazard_ratios", "cause"):
        assert k in r


def test_cause_events(cr_data):
    time, event, X = cr_data
    r = chzrd(time, event, X, cause=1)
    assert r["n_cause_events"] == int(np.sum(event == 1))


def test_hr_positive(cr_data):
    time, event, X = cr_data
    r = chzrd(time, event, X, cause=1)
    assert np.all(r["hazard_ratios"] > 0)


def test_cheatsheet():
    from morie.fn.chzrd import cheatsheet
    assert "cause" in cheatsheet().lower()
