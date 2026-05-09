"""Test opcov."""
import numpy as np
import pytest
from moirais.fn.opcov import opcov


def test_opcov_basic():
    rng = np.random.default_rng(42)
    r = opcov(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opcov_description():
    rng = np.random.default_rng(42)
    r = opcov(n_dims=2, max_iter=50)
    assert r.name
