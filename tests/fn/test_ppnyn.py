"""Test ppnyn."""
import numpy as np
import pytest
from moirais.fn.ppnyn import ppnyn


def test_ppnyn_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppnyn(points=pts, n=30)
    assert r.value is not None


def test_ppnyn_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppnyn(points=pts, n=30)
    assert r.name
