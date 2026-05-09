"""Test jtest."""
import numpy as np
import pytest
from moirais.fn.jtest import jtest


def test_jtest_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = jtest(points=pts, n=40)
    assert r.value is not None


def test_jtest_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = jtest(points=pts, n=40)
    assert r.name
