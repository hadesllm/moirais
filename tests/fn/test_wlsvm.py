"""Test wlsvm."""
import numpy as np
import pytest
from morie.fn.wlsvm import wlsvm


def test_wlsvm_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlsvm(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlsvm_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlsvm(abundance=abund, coords=coords, n=20)
    assert r.name
