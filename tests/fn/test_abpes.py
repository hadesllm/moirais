"""Test abpes."""
import numpy as np
import pytest
from morie.fn.abpes import abpes


def test_abpes_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpes(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abpes_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpes(data=data, coords=coords, n=20)
    assert r.name
