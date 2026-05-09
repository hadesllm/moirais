"""Test ppmc."""
import numpy as np
import pytest
from moirais.fn.ppmc import ppmc


def test_ppmc_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmc(points=pts, n=30)
    assert r.value is not None


def test_ppmc_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmc(points=pts, n=30)
    assert r.name
