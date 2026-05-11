"""Test clafp."""
import numpy as np
import pytest
from morie.fn.clafp import clafp


def test_clafp_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clafp(data=data, n=30, k=3)
    assert r.value is not None


def test_clafp_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clafp(data=data, n=30, k=3)
    assert r.name
