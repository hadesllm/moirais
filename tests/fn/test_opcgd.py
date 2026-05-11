"""Test opcgd."""
import numpy as np
import pytest
from morie.fn.opcgd import opcgd


def test_opcgd_basic():
    rng = np.random.default_rng(42)
    r = opcgd(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opcgd_description():
    rng = np.random.default_rng(42)
    r = opcgd(n_dims=2, max_iter=50)
    assert r.name
