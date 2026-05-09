"""Tests for moirais.fn.clndn -- Concordance index."""

import numpy as np
import pytest

from moirais.fn.clndn import clndn


@pytest.fixture()
def c_data():
    rng = np.random.default_rng(42)
    n = 50
    time = rng.exponential(5, n)
    event = rng.binomial(1, 0.7, n)
    risk = -time + rng.normal(0, 0.5, n)
    return risk, time, event


def test_returns_dict(c_data):
    risk, time, event = c_data
    r = clndn(risk, time, event)
    assert isinstance(r, dict)
    for k in ("c_index", "concordant", "discordant", "tied"):
        assert k in r


def test_c_index_bounded(c_data):
    risk, time, event = c_data
    r = clndn(risk, time, event)
    assert 0 <= r["c_index"] <= 1


def test_perfect_discrimination():
    time = np.array([1, 2, 3, 4, 5.0])
    event = np.ones(5)
    risk = np.array([5, 4, 3, 2, 1.0])
    r = clndn(risk, time, event)
    assert r["c_index"] > 0.9


def test_cheatsheet():
    from moirais.fn.clndn import cheatsheet
    assert "concordance" in cheatsheet().lower()
