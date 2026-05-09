"""Test tssti."""
import numpy as np
import pytest
from moirais.fn.tssti import tssti


def test_tssti_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssti(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssti_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssti(data=data, coords=coords, n=20, t=5)
    assert r.name
