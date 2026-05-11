"""Test abspor."""
import numpy as np
import pytest
from morie.fn.abspor import abspor


def test_abspor_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abspor(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abspor_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abspor(data=data, coords=coords, n=20)
    assert r.name
