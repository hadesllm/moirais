"""Tests for anisotropy ratio."""
import numpy as np
from morie.fn.sganr import sganr


def test_sganr_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    Z = rng.normal(0, 1, 100)
    r = sganr(Z, coords)
    assert r.name == "anisotropy_ratio"
    assert "ratio" in r.extra
    assert "principal_direction" in r.extra


def test_sganr_ratio_positive():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (80, 2))
    Z = rng.normal(0, 1, 80)
    r = sganr(Z, coords)
    assert r.value >= 1.0
