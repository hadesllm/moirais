"""Test clfc."""
import numpy as np
import pytest
from moirais.fn.clfc import clfc


def test_clfc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clfc(data=data, n=30, k=3)
    assert r.value is not None


def test_clfc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clfc(data=data, n=30, k=3)
    assert r.name
