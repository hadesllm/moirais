"""Test ppkno."""
import numpy as np
import pytest
from morie.fn.ppkno import ppkno


def test_ppkno_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppkno(points=pts, n=30)
    assert r.value is not None


def test_ppkno_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppkno(points=pts, n=30)
    assert r.name
