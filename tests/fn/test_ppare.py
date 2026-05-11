"""Test ppare."""
import numpy as np
import pytest
from morie.fn.ppare import ppare


def test_ppare_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppare(points=pts, n=30)
    assert r.value is not None


def test_ppare_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppare(points=pts, n=30)
    assert r.name
