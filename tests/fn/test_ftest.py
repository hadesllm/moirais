"""Test ftest."""
import numpy as np
import pytest
from morie.fn.ftest import ftest


def test_ftest_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = ftest(points=pts, n=40)
    assert r.value is not None


def test_ftest_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = ftest(points=pts, n=40)
    assert r.name
