"""Test opde."""
import numpy as np
import pytest
from moirais.fn.opde import opde


def test_opde_basic():
    rng = np.random.default_rng(42)
    r = opde(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opde_description():
    rng = np.random.default_rng(42)
    r = opde(n_dims=2, max_iter=50)
    assert r.name
