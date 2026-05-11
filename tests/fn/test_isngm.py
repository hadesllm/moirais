"""Test isngm."""
import numpy as np
import pytest
from morie.fn.isngm import isngm


def test_isngm_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = isngm(points=pts, n=40)
    assert r.value is not None


def test_isngm_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = isngm(points=pts, n=40)
    assert r.name
