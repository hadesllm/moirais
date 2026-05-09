"""Test sored."""
import numpy as np
import pytest
from moirais.fn.sored import sored


def test_sored_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sored(data=data, depth=depth, n=20)
    assert r.value is not None


def test_sored_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sored(data=data, depth=depth, n=20)
    assert r.name
