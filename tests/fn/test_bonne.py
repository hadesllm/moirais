"""Test bonne."""
import numpy as np
import pytest
from morie.fn.bonne import bonne


def test_bonne_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = bonne(coords=coords, n=20)
    assert r.value is not None


def test_bonne_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = bonne(coords=coords, n=20)
    assert r.name
