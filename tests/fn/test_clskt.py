"""Test clskt."""
import numpy as np
import pytest
from morie.fn.clskt import clskt


def test_clskt_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clskt(data=data, n=30, k=3)
    assert r.value is not None


def test_clskt_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clskt(data=data, n=30, k=3)
    assert r.name
