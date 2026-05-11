"""Test gnsim."""
import numpy as np
import pytest
from morie.fn.gnsim import gnsim


def test_gnsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = gnsim(points=pts, n=40)
    assert isinstance(r.value, float)
    assert r.value > 0, "Mean nearest-neighbor distance must be positive"
    assert r.value < 100, "Mean NN distance implausibly large for 100x100 window"


def test_gnsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = gnsim(points=pts, n=40)
    assert r.name
