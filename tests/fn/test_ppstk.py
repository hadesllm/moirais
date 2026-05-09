"""Test ppstk."""
import numpy as np
import pytest
from moirais.fn.ppstk import ppstk


def test_ppstk_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstk(points=pts, n=30)
    assert r.value is not None


def test_ppstk_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstk(points=pts, n=30)
    assert r.name
