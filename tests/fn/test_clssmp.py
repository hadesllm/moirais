"""Test clssmp."""
import numpy as np
import pytest
from morie.fn.clssmp import clssmp


def test_clssmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = clssmp(coords=coords, n=20)
    assert r.value is not None


def test_clssmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = clssmp(coords=coords, n=20)
    assert r.name
