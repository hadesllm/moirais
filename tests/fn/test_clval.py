"""Test clval."""
import numpy as np
import pytest
from moirais.fn.clval import clval


def test_clval_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clval(data=data, n=30, k=3)
    assert r.value is not None


def test_clval_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clval(data=data, n=30, k=3)
    assert r.name
