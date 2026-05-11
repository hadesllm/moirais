"""Test dk3bl."""
import numpy as np
import pytest
from morie.fn.dk3bl import dk3bl


def test_dk3bl_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk3bl(x=x, y=y, z=z, values=v, n=15)
    assert r.value is not None


def test_dk3bl_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk3bl(x=x, y=y, z=z, values=v, n=15)
    assert r.name
