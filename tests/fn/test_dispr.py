"""Tests for moirais.fn.dispr -- disparity fit."""

import numpy as np
from moirais.fn.dispr import disparity_fit, dispr


def test_dispr_smoke():
    D_obs = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    D_mod = np.array([[0, 1.1, 2.9], [1.1, 0, 2.1], [2.9, 2.1, 0]], dtype=float)
    r = dispr(D_obs, D_mod)
    assert r.name == "disparity_fit"
    assert "n_pairs" in r.extra


def test_dispr_alias():
    assert dispr is disparity_fit
