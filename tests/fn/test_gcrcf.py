"""Test gcrcf."""
import numpy as np
import pytest
from moirais.fn.gcrcf import gcrcf


def test_gcrcf_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcrcf(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcrcf_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcrcf(data=data, coords=coords, n=30)
    assert r.name
