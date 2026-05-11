"""Test tssdb."""
import numpy as np
import pytest
from morie.fn.tssdb import tssdb


def test_tssdb_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssdb(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssdb_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssdb(data=data, coords=coords, n=20, t=5)
    assert r.name
