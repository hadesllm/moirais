"""Test opsqp."""
import numpy as np
import pytest
from morie.fn.opsqp import opsqp


def test_opsqp_basic():
    rng = np.random.default_rng(42)
    r = opsqp(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opsqp_description():
    rng = np.random.default_rng(42)
    r = opsqp(n_dims=2, max_iter=50)
    assert r.name
