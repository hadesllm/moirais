"""Test ppitj."""
import numpy as np
import pytest
from morie.fn.ppitj import ppitj


def test_ppitj_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitj(points=pts, n=30)
    assert r.value is not None


def test_ppitj_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitj(points=pts, n=30)
    assert r.name
