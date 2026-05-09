"""Test tmpsmp."""
import numpy as np
import pytest
from moirais.fn.tmpsmp import tmpsmp


def test_tmpsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = tmpsmp(coords=coords, n=20)
    assert r.value is not None


def test_tmpsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = tmpsmp(coords=coords, n=20)
    assert r.name
