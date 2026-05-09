"""Test abpmc."""
import numpy as np
import pytest
from moirais.fn.abpmc import abpmc


def test_abpmc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpmc(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abpmc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpmc(data=data, coords=coords, n=20)
    assert r.name
