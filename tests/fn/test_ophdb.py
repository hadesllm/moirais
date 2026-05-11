"""Test ophdb."""
import numpy as np
import pytest
from morie.fn.ophdb import ophdb


def test_ophdb_basic():
    rng = np.random.default_rng(42)
    r = ophdb(n_dims=2, max_iter=50)
    assert r.value is not None


def test_ophdb_description():
    rng = np.random.default_rng(42)
    r = ophdb(n_dims=2, max_iter=50)
    assert r.name
