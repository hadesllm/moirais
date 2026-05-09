"""Test enlnd."""
import numpy as np
import pytest
from moirais.fn.enlnd import enlnd


def test_enlnd_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enlnd(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enlnd_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enlnd(data=data, coords=coords, n=30)
    assert r.name
