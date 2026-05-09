"""Test wlocc."""
import numpy as np
import pytest
from moirais.fn.wlocc import wlocc


def test_wlocc_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlocc(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlocc_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlocc(abundance=abund, coords=coords, n=20)
    assert r.name
