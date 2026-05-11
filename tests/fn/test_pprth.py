"""Test pprth."""
import numpy as np
import pytest
from morie.fn.pprth import pprth


def test_pprth_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pprth(points=pts, n=30)
    assert r.value is not None


def test_pprth_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pprth(points=pts, n=30)
    assert r.name
