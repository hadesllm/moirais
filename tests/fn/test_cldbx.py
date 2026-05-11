"""Test cldbx."""
import numpy as np
import pytest
from morie.fn.cldbx import cldbx


def test_cldbx_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldbx(data=data, n=30, k=3)
    assert r.value is not None


def test_cldbx_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldbx(data=data, n=30, k=3)
    assert r.name
