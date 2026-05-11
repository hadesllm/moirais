"""Test somol."""
import numpy as np
import pytest
from morie.fn.somol import somol


def test_somol_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = somol(data=data, depth=depth, n=20)
    assert r.value is not None


def test_somol_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = somol(data=data, depth=depth, n=20)
    assert r.name
