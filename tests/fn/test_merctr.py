"""Test merctr."""
import numpy as np
import pytest
from morie.fn.merctr import merctr


def test_merctr_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = merctr(coords=coords, n=20)
    assert r.value is not None


def test_merctr_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = merctr(coords=coords, n=20)
    assert r.name
