"""Test opmxm."""
import numpy as np
import pytest
from moirais.fn.opmxm import opmxm


def test_opmxm_basic():
    rng = np.random.default_rng(42)
    r = opmxm(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opmxm_description():
    rng = np.random.default_rng(42)
    r = opmxm(n_dims=2, max_iter=50)
    assert r.name
