"""Tests for moirais.fn.frglt -- Frailty model."""

import numpy as np
import pytest

from moirais.fn.frglt import frglt


@pytest.fixture()
def frailty_data():
    rng = np.random.default_rng(42)
    n = 100
    cluster = np.repeat(np.arange(20), 5)
    X = rng.standard_normal((n, 2))
    lp = 0.5 * X[:, 0]
    t = rng.exponential(np.exp(-lp))
    c = rng.exponential(3, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, X, cluster


def test_returns_dict(frailty_data):
    time, event, X, cluster = frailty_data
    r = frglt(time, event, X, cluster)
    assert isinstance(r, dict)
    for k in ("coefficients", "frailty_variance", "n_clusters"):
        assert k in r


def test_frailty_variance_positive(frailty_data):
    time, event, X, cluster = frailty_data
    r = frglt(time, event, X, cluster)
    assert r["frailty_variance"] > 0


def test_n_clusters(frailty_data):
    time, event, X, cluster = frailty_data
    r = frglt(time, event, X, cluster)
    assert r["n_clusters"] == 20


def test_cheatsheet():
    from moirais.fn.frglt import cheatsheet
    assert "frailty" in cheatsheet().lower()
