"""Test ppitd."""
import numpy as np
import pytest
from moirais.fn.ppitd import ppitd


def test_ppitd_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitd(points=pts, n=30)
    assert r.value is not None


def test_ppitd_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppitd(points=pts, n=30)
    assert r.name
