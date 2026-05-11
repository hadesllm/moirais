"""Test enwsp."""
import numpy as np
import pytest
from morie.fn.enwsp import enwsp


def test_enwsp_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enwsp(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enwsp_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enwsp(data=data, coords=coords, n=30)
    assert r.name
