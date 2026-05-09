"""Test nblmx."""
import numpy as np
import pytest
from moirais.fn.nblmx import nblmx


def test_nblmx_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nblmx(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nblmx_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nblmx(data=data, coords=coords, n=20)
    assert r.name
