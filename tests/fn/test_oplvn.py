"""Test oplvn."""
import numpy as np
import pytest
from morie.fn.oplvn import oplvn


def test_oplvn_basic():
    rng = np.random.default_rng(42)
    r = oplvn(n_dims=2, max_iter=50)
    assert r.value is not None


def test_oplvn_description():
    rng = np.random.default_rng(42)
    r = oplvn(n_dims=2, max_iter=50)
    assert r.name
