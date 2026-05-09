"""Test dk3sk."""
import numpy as np
import pytest
from moirais.fn.dk3sk import dk3sk


def test_dk3sk_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk3sk(x=x, y=y, z=z, values=v, n=15)
    assert r.value is not None


def test_dk3sk_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 15)
    y = rng.uniform(0, 100, 15)
    z = rng.uniform(0, 50, 15)
    v = rng.standard_normal(15)
    r = dk3sk(x=x, y=y, z=z, values=v, n=15)
    assert r.name
