"""Test opcko."""
import numpy as np
import pytest
from morie.fn.opcko import opcko


def test_opcko_basic():
    rng = np.random.default_rng(42)
    r = opcko(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opcko_description():
    rng = np.random.default_rng(42)
    r = opcko(n_dims=2, max_iter=50)
    assert r.name
