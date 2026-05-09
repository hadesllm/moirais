"""Test ppnnd."""
import numpy as np
import pytest
from moirais.fn.ppnnd import ppnnd


def test_ppnnd_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppnnd(points=pts, n=30)
    assert r.value is not None


def test_ppnnd_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppnnd(points=pts, n=30)
    assert r.name
