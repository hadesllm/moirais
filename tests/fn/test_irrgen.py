"""Test irrgen."""
import numpy as np
import pytest
from moirais.fn.irrgen import irrgen


def test_irrgen_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = irrgen(coords=coords, n=20)
    assert r.value is not None


def test_irrgen_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = irrgen(coords=coords, n=20)
    assert r.name
