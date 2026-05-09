"""Test oppar."""
import numpy as np
import pytest
from moirais.fn.oppar import oppar


def test_oppar_basic():
    rng = np.random.default_rng(42)
    r = oppar(n_dims=2, max_iter=50)
    assert r.value is not None


def test_oppar_description():
    rng = np.random.default_rng(42)
    r = oppar(n_dims=2, max_iter=50)
    assert r.name
