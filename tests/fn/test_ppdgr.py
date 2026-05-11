"""Test ppdgr."""
import numpy as np
import pytest
from morie.fn.ppdgr import ppdgr


def test_ppdgr_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppdgr(points=pts, n=30)
    assert r.value is not None


def test_ppdgr_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppdgr(points=pts, n=30)
    assert r.name
