"""Tests for moirais.fn.ufout -- unfolding outliers."""

import numpy as np
from moirais.fn.ufout import unfolding_outliers, ufout


def test_ufout_no_outliers():
    resid = np.array([0.1, -0.1, 0.05, -0.05], dtype=float)
    r = ufout(resid, threshold=2.0)
    assert r.name == "unfolding_outliers"
    assert r.extra["n_outliers"] == 0


def test_ufout_with_outlier():
    resid = np.array([0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1, 0.05, -0.05, 100.0], dtype=float)
    r = ufout(resid, threshold=2.0)
    assert r.extra["n_outliers"] >= 1
    assert 9 in r.value


def test_ufout_alias():
    assert ufout is unfolding_outliers
