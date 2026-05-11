"""Test abpmf."""
import numpy as np
import pytest
from morie.fn.abpmf import abpmf


def test_abpmf_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpmf(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abpmf_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abpmf(data=data, coords=coords, n=20)
    assert r.name
