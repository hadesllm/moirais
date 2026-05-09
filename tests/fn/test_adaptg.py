"""Test adaptg."""
import numpy as np
import pytest
from moirais.fn.adaptg import adaptg


def test_adaptg_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = adaptg(coords=coords, n=20)
    assert r.value is not None


def test_adaptg_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = adaptg(coords=coords, n=20)
    assert r.name
