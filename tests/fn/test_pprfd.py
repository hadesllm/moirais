"""Test pprfd."""
import numpy as np
import pytest
from morie.fn.pprfd import pprfd


def test_pprfd_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pprfd(points=pts, n=30)
    assert r.value is not None


def test_pprfd_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pprfd(points=pts, n=30)
    assert r.name
