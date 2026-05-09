"""Test tssfa."""
import numpy as np
import pytest
from moirais.fn.tssfa import tssfa


def test_tssfa_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssfa(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssfa_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssfa(data=data, coords=coords, n=20, t=5)
    assert r.name
