"""Tests for morie.fn.mssns -- missing sensitivity analysis."""

import numpy as np
from morie.fn.mssns import missing_sensitivity_analysis, mssns


def test_mssns_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 2))
    D = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    r = mssns(D, pcts=[0.1, 0.2], n_trials=2)
    assert r.name == "missing_sensitivity_analysis"
    assert len(r.value["mean_stress"]) == 2


def test_mssns_alias():
    assert mssns is missing_sensitivity_analysis
