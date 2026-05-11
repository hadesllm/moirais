"""Test wlgnf."""
import numpy as np
import pytest
from morie.fn.wlgnf import wlgnf


def test_wlgnf_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlgnf(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlgnf_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlgnf(abundance=abund, coords=coords, n=20)
    assert r.name
