"""Test optrs."""
import numpy as np
import pytest
from morie.fn.optrs import optrs


def test_optrs_basic():
    rng = np.random.default_rng(42)
    r = optrs(n_dims=2, max_iter=50)
    assert r.value is not None


def test_optrs_description():
    rng = np.random.default_rng(42)
    r = optrs(n_dims=2, max_iter=50)
    assert r.name
