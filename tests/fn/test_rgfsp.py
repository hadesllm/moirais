"""Test rgfsp."""
import numpy as np
import pytest
from morie.fn.rgfsp import rgfsp


def test_rgfsp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = rgfsp(x=x, y=y, values=v)
    assert r.value is not None


def test_rgfsp_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = rgfsp(x=x, y=y, values=v)
    assert r.name
