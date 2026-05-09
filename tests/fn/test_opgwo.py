"""Test opgwo."""
import numpy as np
import pytest
from moirais.fn.opgwo import opgwo


def test_opgwo_basic():
    rng = np.random.default_rng(42)
    r = opgwo(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opgwo_description():
    rng = np.random.default_rng(42)
    r = opgwo(n_dims=2, max_iter=50)
    assert r.name
