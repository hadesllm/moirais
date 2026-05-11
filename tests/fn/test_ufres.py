"""Tests for morie.fn.ufres -- unfolding residuals."""

import numpy as np
from morie.fn.ufres import unfolding_residuals, ufres


def test_ufres_smoke():
    obs = np.array([[1, 2], [3, 4]], dtype=float)
    pred = np.array([[1.1, 1.9], [3.2, 3.8]], dtype=float)
    r = ufres(obs, pred)
    assert r.name == "unfolding_residuals"
    assert r.value.shape == (2, 2)
    assert "rmse" in r.extra


def test_ufres_alias():
    assert ufres is unfolding_residuals
