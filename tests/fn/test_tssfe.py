"""Test tssfe."""
import numpy as np
import pytest
from morie.fn.tssfe import tssfe


def test_tssfe_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssfe(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssfe_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssfe(data=data, coords=coords, n=20, t=5)
    assert r.name
