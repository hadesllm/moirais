"""Test clemb."""
import numpy as np
import pytest
from morie.fn.clemb import clemb


def test_clemb_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clemb(data=data, n=30, k=3)
    assert r.value is not None


def test_clemb_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clemb(data=data, n=30, k=3)
    assert r.name
