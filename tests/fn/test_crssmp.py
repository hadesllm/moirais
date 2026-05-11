"""Test crssmp."""
import numpy as np
import pytest
from morie.fn.crssmp import crssmp


def test_crssmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = crssmp(coords=coords, n=20)
    assert r.value is not None


def test_crssmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = crssmp(coords=coords, n=20)
    assert r.name
