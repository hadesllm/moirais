"""Test wlcrc."""
import numpy as np
import pytest
from morie.fn.wlcrc import wlcrc


def test_wlcrc_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlcrc(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlcrc_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlcrc(abundance=abund, coords=coords, n=20)
    assert r.name
