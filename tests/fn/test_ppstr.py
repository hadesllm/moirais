"""Test ppstr."""
import numpy as np
import pytest
from moirais.fn.ppstr import ppstr


def test_ppstr_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstr(points=pts, n=30)
    assert r.value is not None


def test_ppstr_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstr(points=pts, n=30)
    assert r.name
