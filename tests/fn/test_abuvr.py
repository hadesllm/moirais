"""Test abuvr."""
import numpy as np
import pytest
from morie.fn.abuvr import abuvr


def test_abuvr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abuvr(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abuvr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abuvr(data=data, coords=coords, n=20)
    assert r.name
