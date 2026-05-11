"""Test opmfo."""
import numpy as np
import pytest
from morie.fn.opmfo import opmfo


def test_opmfo_basic():
    rng = np.random.default_rng(42)
    r = opmfo(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opmfo_description():
    rng = np.random.default_rng(42)
    r = opmfo(n_dims=2, max_iter=50)
    assert r.name
