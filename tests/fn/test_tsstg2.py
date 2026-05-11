"""Test tsstg2."""
import numpy as np
import pytest
from morie.fn.tsstg2 import tsstg2


def test_tsstg2_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsstg2(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tsstg2_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsstg2(data=data, coords=coords, n=20, t=5)
    assert r.name
