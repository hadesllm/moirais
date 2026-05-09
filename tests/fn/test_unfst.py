"""Tests for moirais.fn.unfst -- unfolding stress diagnostic."""

import numpy as np
from moirais.fn.unfst import unfolding_stress_diagnostic, unfst


def test_unfst_smoke():
    X_resp = np.array([[0, 0], [1, 1]], dtype=float)
    X_stim = np.array([[0.5, 0.5], [2, 2]], dtype=float)
    observed = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            observed[i, j] = np.sqrt(np.sum((X_resp[i] - X_stim[j]) ** 2))
    r = unfst(X_resp, X_stim, observed)
    assert r.name == "unfolding_stress_diagnostic"
    assert r.value < 1e-10


def test_unfst_alias():
    assert unfst is unfolding_stress_diagnostic
