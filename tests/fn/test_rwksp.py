"""Test rwksp."""
import numpy as np
import pytest
from moirais.fn.rwksp import rwksp


def test_rwksp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = rwksp(points=pts, n=40)
    assert r.value is not None


def test_rwksp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = rwksp(points=pts, n=40)
    assert r.name
