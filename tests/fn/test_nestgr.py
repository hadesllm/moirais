"""Test nestgr."""
import numpy as np
import pytest
from morie.fn.nestgr import nestgr


def test_nestgr_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = nestgr(coords=coords, n=20)
    assert r.value is not None


def test_nestgr_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = nestgr(coords=coords, n=20)
    assert r.name
