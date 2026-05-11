"""Test ppipp."""
import numpy as np
import pytest
from morie.fn.ppipp import ppipp


def test_ppipp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppipp(points=pts, n=30)
    assert r.value is not None


def test_ppipp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppipp(points=pts, n=30)
    assert r.name
