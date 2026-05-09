"""Test opred."""
import numpy as np
import pytest
from moirais.fn.opred import opred


def test_opred_basic():
    rng = np.random.default_rng(42)
    r = opred(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opred_description():
    rng = np.random.default_rng(42)
    r = opred(n_dims=2, max_iter=50)
    assert r.name
