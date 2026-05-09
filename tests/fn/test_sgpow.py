"""Tests for power variogram."""
import numpy as np
from moirais.fn.sgpow import sgpow


def test_sgpow_smoke():
    h = np.array([0, 1, 2, 3], dtype=float)
    r = sgpow(h, c0=0.1, c1=0.5, alpha=1.5)
    assert r.name == "power_variogram"
    assert r.extra["gamma"][0] == 0.0
    assert r.extra["gamma"][1] > 0


def test_sgpow_invalid_alpha():
    import pytest
    with pytest.raises(ValueError):
        sgpow(np.array([1.0]), c0=0, c1=1, alpha=2.5)
