"""Test snsim."""
import numpy as np
import pytest
from morie.fn.snsim import snsim


def test_snsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = snsim(points=pts, n=40)
    assert r.value is not None


def test_snsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = snsim(points=pts, n=40)
    assert r.name
