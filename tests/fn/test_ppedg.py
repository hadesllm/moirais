"""Test ppedg."""
import numpy as np
import pytest
from moirais.fn.ppedg import ppedg


def test_ppedg_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppedg(points=pts, n=30)
    assert r.value is not None


def test_ppedg_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppedg(points=pts, n=30)
    assert r.name
