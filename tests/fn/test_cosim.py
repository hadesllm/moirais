"""Test cosim."""
import numpy as np
import pytest
from moirais.fn.cosim import cosim


def test_cosim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = cosim(points=pts, n=40)
    assert r.value is not None


def test_cosim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = cosim(points=pts, n=40)
    assert r.name
