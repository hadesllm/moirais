"""Test clflm."""
import numpy as np
import pytest
from moirais.fn.clflm import clflm


def test_clflm_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clflm(data=data, n=30, k=3)
    assert r.value is not None


def test_clflm_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clflm(data=data, n=30, k=3)
    assert r.name
