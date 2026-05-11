"""Test vorare."""
import numpy as np
import pytest
from morie.fn.vorare import vorare


def test_vorare_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vorare(x=x, y=y, values=v)
    assert r.value is not None


def test_vorare_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vorare(x=x, y=y, values=v)
    assert r.name
