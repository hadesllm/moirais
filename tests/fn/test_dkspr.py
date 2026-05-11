"""Test dkspr."""
import numpy as np
import pytest
from morie.fn.dkspr import dkspr


def test_dkspr_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dkspr(x=x, y=y, z=z, values=v, n=15)
    assert r.value is not None


def test_dkspr_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dkspr(x=x, y=y, z=z, values=v, n=15)
    assert r.name
