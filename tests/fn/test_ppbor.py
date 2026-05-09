"""Test ppbor."""
import numpy as np
import pytest
from moirais.fn.ppbor import ppbor


def test_ppbor_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppbor(points=pts, n=30)
    assert r.value is not None


def test_ppbor_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppbor(points=pts, n=30)
    assert r.name
