"""Test clclr."""
import numpy as np
import pytest
from moirais.fn.clclr import clclr


def test_clclr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clclr(data=data, n=30, k=3)
    assert r.value is not None


def test_clclr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clclr(data=data, n=30, k=3)
    assert r.name
