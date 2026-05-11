"""Test datumx."""
import numpy as np
import pytest
from morie.fn.datumx import datumx


def test_datumx_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = datumx(coords=coords, n=20)
    assert r.value is not None


def test_datumx_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = datumx(coords=coords, n=20)
    assert r.name
