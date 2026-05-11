"""Tests for intrinsic CAR model."""
import numpy as np
from morie.fn.sgicar import sgicar


def test_sgicar_smoke():
    n = 8
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            if i != j:
                W[i, j] = 1.0
    Z = np.arange(n, dtype=float)
    r = sgicar(Z, W)
    assert r.name == "intrinsic_car_model"
    assert r.statistic >= 0
    assert "conditional_means" in r.extra


def test_cheatsheet():
    from morie.fn.sgicar import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
