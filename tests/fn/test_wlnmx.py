"""Test wlnmx."""
import numpy as np
import pytest
from morie.fn.wlnmx import wlnmx


def test_wlnmx_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlnmx(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlnmx_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlnmx(abundance=abund, coords=coords, n=20)
    assert r.name
