"""Test havrsi."""
import numpy as np
import pytest
from moirais.fn.havrsi import havrsi


def test_havrsi_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = havrsi(coords=coords, n=20)
    assert r.value is not None


def test_havrsi_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = havrsi(coords=coords, n=20)
    assert r.name
