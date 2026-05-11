"""Test mhsim."""
import numpy as np
import pytest
from morie.fn.mhsim import mhsim


def test_mhsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = mhsim(points=pts, n=40)
    assert r.value is not None


def test_mhsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = mhsim(points=pts, n=40)
    assert r.name
