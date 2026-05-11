"""Test blksmp."""
import numpy as np
import pytest
from morie.fn.blksmp import blksmp


def test_blksmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = blksmp(coords=coords, n=20)
    assert r.value is not None


def test_blksmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = blksmp(coords=coords, n=20)
    assert r.name
