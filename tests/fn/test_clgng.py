"""Test clgng."""
import numpy as np
import pytest
from morie.fn.clgng import clgng


def test_clgng_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clgng(data=data, n=30, k=3)
    assert r.value is not None


def test_clgng_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clgng(data=data, n=30, k=3)
    assert r.name
