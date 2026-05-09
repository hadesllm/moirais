"""Test mollwd."""
import numpy as np
import pytest
from moirais.fn.mollwd import mollwd


def test_mollwd_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = mollwd(coords=coords, n=20)
    assert r.value is not None


def test_mollwd_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = mollwd(coords=coords, n=20)
    assert r.name
