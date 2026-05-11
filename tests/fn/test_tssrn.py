"""Test tssrn."""
import numpy as np
import pytest
from morie.fn.tssrn import tssrn


def test_tssrn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssrn(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssrn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssrn(data=data, coords=coords, n=20, t=5)
    assert r.name
