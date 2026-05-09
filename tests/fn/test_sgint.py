"""Tests for intensity estimation."""
import numpy as np
from moirais.fn.sgint import sgint


def test_sgint_kernel():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (50, 2))
    r = sgint(pts, (0, 10, 0, 10), method="kernel", grid_n=20)
    assert r.name == "intensity_estimate"
    assert r.extra["global_intensity"] > 0


def test_sgint_quadrat():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (50, 2))
    r = sgint(pts, (0, 10, 0, 10), method="quadrat", grid_n=25)
    assert r.name == "intensity_estimate"
