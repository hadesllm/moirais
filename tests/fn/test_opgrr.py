"""Test opgrr."""
import numpy as np
import pytest
from moirais.fn.opgrr import opgrr


def test_opgrr_basic():
    rng = np.random.default_rng(42)
    r = opgrr(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opgrr_description():
    rng = np.random.default_rng(42)
    r = opgrr(n_dims=2, max_iter=50)
    assert r.name
