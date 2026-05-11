"""Test clreg."""
import numpy as np
import pytest
from morie.fn.clreg import clreg


def test_clreg_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clreg(data=data, n=30, k=3)
    assert r.value is not None


def test_clreg_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clreg(data=data, n=30, k=3)
    assert r.name
