"""Test rhumbr."""
import numpy as np
import pytest
from morie.fn.rhumbr import rhumbr


def test_rhumbr_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = rhumbr(coords=coords, n=20)
    assert r.value is not None


def test_rhumbr_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = rhumbr(coords=coords, n=20)
    assert r.name
