"""Test ppbay."""
import numpy as np
import pytest
from moirais.fn.ppbay import ppbay


def test_ppbay_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppbay(points=pts, n=30)
    assert r.value is not None


def test_ppbay_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppbay(points=pts, n=30)
    assert r.name
