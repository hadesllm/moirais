"""Test cksim."""
import numpy as np
import pytest
from moirais.fn.cksim import cksim


def test_cksim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = cksim(points=pts, n=40)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value > 0


def test_cksim_extra():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = cksim(points=pts, n=40)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert r.extra["n"] == 40
    assert r.extra["mean_nn_dist"] > 0
