"""Test absbs."""
import numpy as np
import pytest
from moirais.fn.absbs import absbs


def test_absbs_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = absbs(data=data, coords=coords, n=20)
    assert r.value is not None


def test_absbs_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = absbs(data=data, coords=coords, n=20)
    assert r.name
