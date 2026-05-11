"""Test dk4sm."""
import numpy as np
import pytest
from morie.fn.dk4sm import dk4sm


def test_dk4sm_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk4sm(x=x, y=y, z=z, values=v, n=15)
    assert r.value is not None


def test_dk4sm_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk4sm(x=x, y=y, z=z, values=v, n=15)
    assert r.name
