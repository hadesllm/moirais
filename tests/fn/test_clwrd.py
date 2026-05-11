"""Test clwrd."""
import numpy as np
import pytest
from morie.fn.clwrd import clwrd


def test_clwrd_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clwrd(data=data, n=30, k=3)
    assert r.value is not None


def test_clwrd_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clwrd(data=data, n=30, k=3)
    assert r.name
