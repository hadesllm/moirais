"""Test clred."""
import numpy as np
import pytest
from morie.fn.clred import clred


def test_clred_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clred(data=data, n=30, k=3)
    assert r.value is not None


def test_clred_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clred(data=data, n=30, k=3)
    assert r.name
