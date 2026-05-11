"""Test mls."""
import numpy as np
import pytest
from morie.fn.mls import mls


def test_mls_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = mls(x=x, y=y, values=v)
    assert r.value is not None


def test_mls_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = mls(x=x, y=y, values=v)
    assert r.name
