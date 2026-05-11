"""Test enlgr."""
import numpy as np
import pytest
from morie.fn.enlgr import enlgr


def test_enlgr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enlgr(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enlgr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enlgr(data=data, coords=coords, n=30)
    assert r.name
