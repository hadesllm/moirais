"""Test ppstg."""
import numpy as np
import pytest
from moirais.fn.ppstg import ppstg


def test_ppstg_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstg(points=pts, n=30)
    assert r.value is not None


def test_ppstg_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppstg(points=pts, n=30)
    assert r.name
