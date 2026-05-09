"""Test clsng."""
import numpy as np
import pytest
from moirais.fn.clsng import clsng


def test_clsng_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsng(data=data, n=30, k=3)
    assert r.value is not None


def test_clsng_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsng(data=data, n=30, k=3)
    assert r.name
