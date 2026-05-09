"""Test clsil."""
import numpy as np
import pytest
from moirais.fn.clsil import clsil


def test_clsil_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsil(data=data, n=30, k=3)
    assert r.value is not None


def test_clsil_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsil(data=data, n=30, k=3)
    assert r.name
