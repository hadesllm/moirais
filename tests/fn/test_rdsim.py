"""Test rdsim."""
import numpy as np
import pytest
from morie.fn.rdsim import rdsim


def test_rdsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = rdsim(points=pts, n=40)
    assert r.value is not None


def test_rdsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = rdsim(points=pts, n=40)
    assert r.name
