"""Test dssim."""
import numpy as np
import pytest
from moirais.fn.dssim import dssim


def test_dssim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = dssim(points=pts, n=40)
    assert r.value is not None


def test_dssim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = dssim(points=pts, n=40)
    assert r.name
