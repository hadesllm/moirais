"""Test cassni."""
import numpy as np
import pytest
from morie.fn.cassni import cassni


def test_cassni_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = cassni(coords=coords, n=20)
    assert r.value is not None


def test_cassni_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = cassni(coords=coords, n=20)
    assert r.name
