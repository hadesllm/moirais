"""Test optsp."""
import numpy as np
import pytest
from morie.fn.optsp import optsp


def test_optsp_basic():
    rng = np.random.default_rng(42)
    r = optsp(n_dims=2, max_iter=50)
    assert r.value is not None


def test_optsp_description():
    rng = np.random.default_rng(42)
    r = optsp(n_dims=2, max_iter=50)
    assert r.name
