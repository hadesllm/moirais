"""Tests for moirais.fn.srmdc -- Survival model discrimination."""

import numpy as np
import pytest

from moirais.fn.srmdc import srmdc


@pytest.fixture()
def disc_data():
    rng = np.random.default_rng(42)
    n = 50
    time = rng.exponential(5, n)
    event = rng.binomial(1, 0.7, n)
    risk = -time + rng.normal(0, 0.5, n)
    return risk, time, event


def test_returns_dict(disc_data):
    risk, time, event = disc_data
    r = srmdc(risk, time, event)
    assert isinstance(r, dict)
    for k in ("c_index", "d_statistic", "auc_values"):
        assert k in r


def test_c_index_bounded(disc_data):
    risk, time, event = disc_data
    r = srmdc(risk, time, event)
    assert 0 <= r["c_index"] <= 1


def test_auc_bounded(disc_data):
    risk, time, event = disc_data
    r = srmdc(risk, time, event)
    assert np.all(np.array(r["auc_values"]) >= 0)
    assert np.all(np.array(r["auc_values"]) <= 1)


def test_cheatsheet():
    from moirais.fn.srmdc import cheatsheet
    assert "discrimination" in cheatsheet().lower()
