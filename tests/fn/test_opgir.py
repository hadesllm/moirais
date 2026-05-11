"""Test opgir."""
import numpy as np
import pytest
from morie.fn.opgir import opgir


def test_opgir_basic():
    rng = np.random.default_rng(42)
    r = opgir(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opgir_description():
    rng = np.random.default_rng(42)
    r = opgir(n_dims=2, max_iter=50)
    assert r.name
