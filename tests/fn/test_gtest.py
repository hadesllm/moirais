"""Test gtest."""
import numpy as np
import pytest
from moirais.fn.gtest import gtest


def test_gtest_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = gtest(points=pts, n=40)
    assert r.value is not None


def test_gtest_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = gtest(points=pts, n=40)
    assert r.name
