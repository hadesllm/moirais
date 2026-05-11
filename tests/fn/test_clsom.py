"""Test clsom."""
import numpy as np
import pytest
from morie.fn.clsom import clsom


def test_clsom_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsom(data=data, n=30, k=3)
    assert r.value is not None


def test_clsom_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsom(data=data, n=30, k=3)
    assert r.name
