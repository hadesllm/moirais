"""Test dk4vg."""
import numpy as np
import pytest
from moirais.fn.dk4vg import dk4vg


def test_dk4vg_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk4vg(x=x, y=y, z=z, values=v, n=15)
    assert r.value is not None


def test_dk4vg_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk4vg(x=x, y=y, z=z, values=v, n=15)
    assert r.name
