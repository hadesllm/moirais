"""Test ppstl."""
import numpy as np
import pytest
from morie.fn.ppstl import ppstl


def test_ppstl_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstl(points=pts, n=30)
    assert r.value is not None


def test_ppstl_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstl(points=pts, n=30)
    assert r.name
