"""Test clcmp."""
import numpy as np
import pytest
from morie.fn.clcmp import clcmp


def test_clcmp_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcmp(data=data, n=30, k=3)
    assert r.value is not None


def test_clcmp_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcmp(data=data, n=30, k=3)
    assert r.name
