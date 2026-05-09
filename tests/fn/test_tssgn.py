"""Test tssgn."""
import numpy as np
import pytest
from moirais.fn.tssgn import tssgn


def test_tssgn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssgn(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssgn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssgn(data=data, coords=coords, n=20, t=5)
    assert r.name
