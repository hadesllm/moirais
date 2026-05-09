"""Test wldst."""
import numpy as np
import pytest
from moirais.fn.wldst import wldst


def test_wldst_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wldst(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wldst_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wldst(abundance=abund, coords=coords, n=20)
    assert r.name
