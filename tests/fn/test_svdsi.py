"""Test svdsi."""
import numpy as np
import pytest
from morie.fn.svdsi import svdsi


def test_svdsi_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = svdsi(points=pts, n=40)
    assert r.value is not None


def test_svdsi_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = svdsi(points=pts, n=40)
    assert r.name
