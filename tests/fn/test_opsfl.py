"""Test opsfl."""
import numpy as np
import pytest
from moirais.fn.opsfl import opsfl


def test_opsfl_basic():
    rng = np.random.default_rng(42)
    r = opsfl(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opsfl_description():
    rng = np.random.default_rng(42)
    r = opsfl(n_dims=2, max_iter=50)
    assert r.name
