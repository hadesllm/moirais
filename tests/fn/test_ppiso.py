"""Test ppiso."""
import numpy as np
import pytest
from moirais.fn.ppiso import ppiso


def test_ppiso_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppiso(points=pts, n=30)
    assert r.value is not None


def test_ppiso_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppiso(points=pts, n=30)
    assert r.name
