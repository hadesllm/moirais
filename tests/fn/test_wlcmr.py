"""Test wlcmr."""
import numpy as np
import pytest
from moirais.fn.wlcmr import wlcmr


def test_wlcmr_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlcmr(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlcmr_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlcmr(abundance=abund, coords=coords, n=20)
    assert r.name
