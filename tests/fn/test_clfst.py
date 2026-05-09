"""Test clfst."""
import numpy as np
import pytest
from moirais.fn.clfst import clfst


def test_clfst_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clfst(data=data, n=30, k=3)
    assert r.value is not None


def test_clfst_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clfst(data=data, n=30, k=3)
    assert r.name
