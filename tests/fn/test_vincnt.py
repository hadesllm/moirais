"""Test vincnt."""
import numpy as np
import pytest
from morie.fn.vincnt import vincnt


def test_vincnt_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = vincnt(coords=coords, n=20)
    assert r.value is not None


def test_vincnt_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = vincnt(coords=coords, n=20)
    assert r.name
