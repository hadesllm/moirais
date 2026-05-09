"""Test clsta."""
import numpy as np
import pytest
from moirais.fn.clsta import clsta


def test_clsta_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsta(data=data, n=30, k=3)
    assert r.value is not None


def test_clsta_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clsta(data=data, n=30, k=3)
    assert r.name
