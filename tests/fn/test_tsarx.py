"""Test tsarx."""
import numpy as np
import pytest
from morie.fn.tsarx import tsarx


def test_tsarx_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsarx(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tsarx_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsarx(data=data, coords=coords, n=20, t=5)
    assert r.name
