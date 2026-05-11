"""Test clcon."""
import numpy as np
import pytest
from morie.fn.clcon import clcon


def test_clcon_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcon(data=data, n=30, k=3)
    assert r.value is not None


def test_clcon_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcon(data=data, n=30, k=3)
    assert r.name
