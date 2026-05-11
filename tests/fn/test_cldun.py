"""Test cldun."""
import numpy as np
import pytest
from morie.fn.cldun import cldun


def test_cldun_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldun(data=data, n=30, k=3)
    assert r.value is not None


def test_cldun_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldun(data=data, n=30, k=3)
    assert r.name
