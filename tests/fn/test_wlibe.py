"""Test wlibe."""
import numpy as np
import pytest
from morie.fn.wlibe import wlibe


def test_wlibe_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlibe(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlibe_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlibe(abundance=abund, coords=coords, n=20)
    assert r.name
