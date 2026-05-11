"""Test clspc."""
import numpy as np
import pytest
from morie.fn.clspc import clspc


def test_clspc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clspc(data=data, n=30, k=3)
    assert r.value is not None


def test_clspc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clspc(data=data, n=30, k=3)
    assert r.name
