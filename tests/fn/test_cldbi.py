"""Test cldbi."""
import numpy as np
import pytest
from morie.fn.cldbi import cldbi


def test_cldbi_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldbi(data=data, n=30, k=3)
    assert r.value is not None


def test_cldbi_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldbi(data=data, n=30, k=3)
    assert r.name
