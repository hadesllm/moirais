"""Test opsa."""
import numpy as np
import pytest
from moirais.fn.opsa import opsa


def test_opsa_basic():
    rng = np.random.default_rng(42)
    r = opsa(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opsa_description():
    rng = np.random.default_rng(42)
    r = opsa(n_dims=2, max_iter=50)
    assert r.name
