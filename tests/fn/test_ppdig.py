"""Test ppdig."""
import numpy as np
import pytest
from moirais.fn.ppdig import ppdig


def test_ppdig_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppdig(points=pts, n=30)
    assert r.value is not None


def test_ppdig_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppdig(points=pts, n=30)
    assert r.name
