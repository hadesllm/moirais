"""Test clhdb."""
import numpy as np
import pytest
from moirais.fn.clhdb import clhdb


def test_clhdb_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clhdb(data=data, n=30, k=3)
    assert r.value is not None


def test_clhdb_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clhdb(data=data, n=30, k=3)
    assert r.name
