"""Test abpfc."""
import numpy as np
import pytest
from morie.fn.abpfc import abpfc


def test_abpfc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpfc(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abpfc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpfc(data=data, coords=coords, n=20)
    assert r.name
