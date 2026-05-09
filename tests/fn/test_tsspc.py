"""Test tsspc."""
import numpy as np
import pytest
from moirais.fn.tsspc import tsspc


def test_tsspc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsspc(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tsspc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsspc(data=data, coords=coords, n=20, t=5)
    assert r.name
