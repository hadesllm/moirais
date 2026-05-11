"""Test gcstr."""
import numpy as np
import pytest
from morie.fn.gcstr import gcstr


def test_gcstr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcstr(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcstr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcstr(data=data, coords=coords, n=30)
    assert r.name
