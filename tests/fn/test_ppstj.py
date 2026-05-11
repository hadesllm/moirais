"""Test ppstj."""
import numpy as np
import pytest
from morie.fn.ppstj import ppstj


def test_ppstj_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstj(points=pts, n=30)
    assert r.value is not None


def test_ppstj_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstj(points=pts, n=30)
    assert r.name
