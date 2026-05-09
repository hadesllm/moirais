"""Test soper."""
import numpy as np
import pytest
from moirais.fn.soper import soper


def test_soper_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soper(data=data, depth=depth, n=20)
    assert r.value is not None


def test_soper_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soper(data=data, depth=depth, n=20)
    assert r.name
