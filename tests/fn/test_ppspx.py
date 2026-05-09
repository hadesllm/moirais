"""Test ppspx."""
import numpy as np
import pytest
from moirais.fn.ppspx import ppspx


def test_ppspx_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppspx(points=pts, n=30)
    assert r.value is not None


def test_ppspx_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppspx(points=pts, n=30)
    assert r.name
