"""Test opnwt."""
import numpy as np
import pytest
from moirais.fn.opnwt import opnwt


def test_opnwt_basic():
    rng = np.random.default_rng(42)
    r = opnwt(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opnwt_description():
    rng = np.random.default_rng(42)
    r = opnwt(n_dims=2, max_iter=50)
    assert r.name
