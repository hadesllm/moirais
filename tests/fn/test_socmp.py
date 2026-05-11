"""Test socmp."""
import numpy as np
import pytest
from morie.fn.socmp import socmp


def test_socmp_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = socmp(data=data, depth=depth, n=20)
    assert r.value is not None


def test_socmp_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = socmp(data=data, depth=depth, n=20)
    assert r.name
