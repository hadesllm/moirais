"""Test clazp."""
import numpy as np
import pytest
from moirais.fn.clazp import clazp


def test_clazp_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clazp(data=data, n=30, k=3)
    assert r.value is not None


def test_clazp_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clazp(data=data, n=30, k=3)
    assert r.name
