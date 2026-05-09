"""Test mpsim."""
import numpy as np
import pytest
from moirais.fn.mpsim import mpsim


def test_mpsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = mpsim(points=pts, n=40)
    assert r.value is not None


def test_mpsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = mpsim(points=pts, n=40)
    assert r.name
