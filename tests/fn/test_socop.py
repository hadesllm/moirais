"""Test socop."""
import numpy as np
import pytest
from morie.fn.socop import socop


def test_socop_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = socop(data=data, depth=depth, n=20)
    assert r.value is not None


def test_socop_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = socop(data=data, depth=depth, n=20)
    assert r.name
