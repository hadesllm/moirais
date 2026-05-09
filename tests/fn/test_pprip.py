"""Test pprip."""
import numpy as np
import pytest
from moirais.fn.pprip import pprip


def test_pprip_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pprip(points=pts, n=30)
    assert r.value is not None


def test_pprip_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pprip(points=pts, n=30)
    assert r.name
