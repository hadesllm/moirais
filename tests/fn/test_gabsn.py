"""Test gabsn."""
import numpy as np
import pytest
from morie.fn.gabsn import gabsn


def test_gabsn_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = gabsn(x=x, y=y, values=v)
    assert r.value is not None


def test_gabsn_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = gabsn(x=x, y=y, values=v)
    assert r.name
