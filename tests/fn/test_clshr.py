"""Test clshr."""
import numpy as np
import pytest
from moirais.fn.clshr import clshr


def test_clshr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clshr(data=data, n=30, k=3)
    assert r.value is not None


def test_clshr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clshr(data=data, n=30, k=3)
    assert r.name
