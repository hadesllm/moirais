"""Test trigen."""
import numpy as np
import pytest
from moirais.fn.trigen import trigen


def test_trigen_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = trigen(coords=coords, n=20)
    assert r.value is not None


def test_trigen_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = trigen(coords=coords, n=20)
    assert r.name
