"""Test clden."""
import numpy as np
import pytest
from moirais.fn.clden import clden


def test_clden_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clden(data=data, n=30, k=3)
    assert r.value is not None


def test_clden_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clden(data=data, n=30, k=3)
    assert r.name
