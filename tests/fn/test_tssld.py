"""Test tssld."""
import numpy as np
import pytest
from moirais.fn.tssld import tssld


def test_tssld_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssld(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssld_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssld(data=data, coords=coords, n=20, t=5)
    assert r.name
