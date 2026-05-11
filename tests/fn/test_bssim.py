"""Test bssim."""
import numpy as np
import pytest
from morie.fn.bssim import bssim


def test_bssim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = bssim(points=pts, n=40)
    assert r.value is not None


def test_bssim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = bssim(points=pts, n=40)
    assert r.name
