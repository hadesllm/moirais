"""Test curvgr."""
import numpy as np
import pytest
from morie.fn.curvgr import curvgr


def test_curvgr_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = curvgr(coords=coords, n=20)
    assert r.value is not None


def test_curvgr_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = curvgr(coords=coords, n=20)
    assert r.name
