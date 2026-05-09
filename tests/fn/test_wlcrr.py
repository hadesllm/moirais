"""Test wlcrr."""
import numpy as np
import pytest
from moirais.fn.wlcrr import wlcrr


def test_wlcrr_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlcrr(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlcrr_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlcrr(abundance=abund, coords=coords, n=20)
    assert r.name
