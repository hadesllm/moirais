"""Test clpfc."""
import numpy as np
import pytest
from moirais.fn.clpfc import clpfc


def test_clpfc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clpfc(data=data, n=30, k=3)
    assert r.value is not None


def test_clpfc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clpfc(data=data, n=30, k=3)
    assert r.name
