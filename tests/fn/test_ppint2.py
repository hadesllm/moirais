"""Test ppint2."""
import numpy as np
import pytest
from morie.fn.ppint2 import ppint2


def test_ppint2_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppint2(points=pts, n=30)
    assert r.value is not None


def test_ppint2_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppint2(points=pts, n=30)
    assert r.name
