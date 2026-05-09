"""Test hexgen."""
import numpy as np
import pytest
from moirais.fn.hexgen import hexgen


def test_hexgen_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = hexgen(coords=coords, n=20)
    assert r.value is not None


def test_hexgen_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = hexgen(coords=coords, n=20)
    assert r.name
