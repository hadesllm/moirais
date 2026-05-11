"""Test soagg."""
import numpy as np
import pytest
from morie.fn.soagg import soagg


def test_soagg_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soagg(data=data, depth=depth, n=20)
    assert r.value is not None


def test_soagg_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soagg(data=data, depth=depth, n=20)
    assert r.name
