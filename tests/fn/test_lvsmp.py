"""Test lvsmp."""
import numpy as np
import pytest
from morie.fn.lvsmp import lvsmp


def test_lvsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = lvsmp(coords=coords, n=20)
    assert r.value is not None


def test_lvsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = lvsmp(coords=coords, n=20)
    assert r.name
