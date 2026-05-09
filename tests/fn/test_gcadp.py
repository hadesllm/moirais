"""Test gcadp."""
import numpy as np
import pytest
from moirais.fn.gcadp import gcadp


def test_gcadp_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcadp(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcadp_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcadp(data=data, coords=coords, n=30)
    assert r.name
