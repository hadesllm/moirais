"""Test robnsn."""
import numpy as np
import pytest
from morie.fn.robnsn import robnsn


def test_robnsn_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = robnsn(coords=coords, n=20)
    assert r.value is not None


def test_robnsn_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = robnsn(coords=coords, n=20)
    assert r.name
