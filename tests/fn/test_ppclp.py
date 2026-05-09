"""Test ppclp."""
import numpy as np
import pytest
from moirais.fn.ppclp import ppclp


def test_ppclp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppclp(points=pts, n=30)
    assert r.value is not None


def test_ppclp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppclp(points=pts, n=30)
    assert r.name
