"""Test ltest."""
import numpy as np
import pytest
from moirais.fn.ltest import ltest


def test_ltest_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = ltest(points=pts, n=40)
    assert r.value is not None


def test_ltest_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = ltest(points=pts, n=40)
    assert r.name
