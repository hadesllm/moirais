"""Test ppsts."""
import numpy as np
import pytest
from morie.fn.ppsts import ppsts


def test_ppsts_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppsts(points=pts, n=30)
    assert r.value is not None


def test_ppsts_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppsts(points=pts, n=30)
    assert r.name
