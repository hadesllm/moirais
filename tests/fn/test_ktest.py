"""Test ktest."""
import numpy as np
import pytest
from moirais.fn.ktest import ktest


def test_ktest_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = ktest(points=pts, n=40)
    assert r.value is not None


def test_ktest_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = ktest(points=pts, n=40)
    assert r.name
