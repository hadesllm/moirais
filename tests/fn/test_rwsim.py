"""Test rwsim."""
import numpy as np
import pytest
from morie.fn.rwsim import rwsim


def test_rwsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = rwsim(points=pts, n=40)
    assert r.value is not None


def test_rwsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = rwsim(points=pts, n=40)
    assert r.name
