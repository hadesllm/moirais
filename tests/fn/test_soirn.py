"""Test soirn."""
import numpy as np
import pytest
from morie.fn.soirn import soirn


def test_soirn_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soirn(data=data, depth=depth, n=20)
    assert r.value is not None


def test_soirn_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soirn(data=data, depth=depth, n=20)
    assert r.name
