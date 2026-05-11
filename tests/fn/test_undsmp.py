"""Test undsmp."""
import numpy as np
import pytest
from morie.fn.undsmp import undsmp


def test_undsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = undsmp(coords=coords, n=20)
    assert r.value is not None


def test_undsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = undsmp(coords=coords, n=20)
    assert r.name
