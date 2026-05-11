"""Tests for morie.fn.ufr2 -- unfolding R-squared."""

import numpy as np
from morie.fn.ufr2 import unfolding_r_squared, ufr2


def test_ufr2_perfect():
    obs = np.array([1, 2, 3, 4, 5], dtype=float)
    r = ufr2(obs, obs)
    assert r.name == "unfolding_r_squared"
    assert np.isclose(r.value, 1.0)


def test_ufr2_imperfect():
    obs = np.array([1, 2, 3, 4, 5], dtype=float)
    pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2], dtype=float)
    r = ufr2(obs, pred)
    assert 0 < r.value < 1


def test_ufr2_alias():
    assert ufr2 is unfolding_r_squared
