"""Tests for lognormal kriging."""
import numpy as np
from morie.fn.sglnk import sglnk


def test_sglnk_smoke():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    Z = np.array([1.0, 2.0, 3.0, 4.0])
    r = sglnk(Z, coords, np.array([0.5, 0.5]))
    assert r.name == "lognormal_kriging"
    assert r.statistic > 0
    assert "log_prediction" in r.extra


def test_sglnk_rejects_negative():
    import pytest
    coords = np.array([[0, 0], [1, 0]], dtype=float)
    Z = np.array([-1.0, 2.0])
    with pytest.raises(ValueError):
        sglnk(Z, coords, np.array([0.5, 0.5]))
