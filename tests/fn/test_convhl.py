"""Test convhl."""
import numpy as np
import pytest
from morie.fn.convhl import convhl


def test_convhl_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = convhl(x=x, y=y, values=v)
    assert r.value is not None


def test_convhl_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = convhl(x=x, y=y, values=v)
    assert r.name
