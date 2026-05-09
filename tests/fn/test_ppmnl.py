"""Test ppmnl."""
import numpy as np
import pytest
from moirais.fn.ppmnl import ppmnl


def test_ppmnl_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmnl(points=pts, n=30)
    assert r.value is not None


def test_ppmnl_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmnl(points=pts, n=30)
    assert r.name
