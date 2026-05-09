"""Test clmnb."""
import numpy as np
import pytest
from moirais.fn.clmnb import clmnb


def test_clmnb_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clmnb(data=data, n=30, k=3)
    assert r.value is not None


def test_clmnb_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clmnb(data=data, n=30, k=3)
    assert r.name
