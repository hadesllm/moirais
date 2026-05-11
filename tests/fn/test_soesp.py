"""Test soesp."""
import numpy as np
import pytest
from morie.fn.soesp import soesp


def test_soesp_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soesp(data=data, depth=depth, n=20)
    assert r.value is not None


def test_soesp_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soesp(data=data, depth=depth, n=20)
    assert r.name
