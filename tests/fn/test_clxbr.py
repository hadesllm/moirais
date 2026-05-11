"""Test clxbr."""
import numpy as np
import pytest
from morie.fn.clxbr import clxbr


def test_clxbr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clxbr(data=data, n=30, k=3)
    assert r.value is not None


def test_clxbr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clxbr(data=data, n=30, k=3)
    assert r.name
