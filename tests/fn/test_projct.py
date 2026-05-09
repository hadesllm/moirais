"""Test projct."""
import numpy as np
import pytest
from moirais.fn.projct import projct


def test_projct_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = projct(coords=coords, n=20)
    assert r.value is not None


def test_projct_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = projct(coords=coords, n=20)
    assert r.name
