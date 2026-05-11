"""Test abblc."""
import numpy as np
import pytest
from morie.fn.abblc import abblc


def test_abblc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abblc(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abblc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abblc(data=data, coords=coords, n=20)
    assert r.name
