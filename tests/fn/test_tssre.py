"""Test tssre."""
import numpy as np
import pytest
from morie.fn.tssre import tssre


def test_tssre_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssre(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tssre_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tssre(data=data, coords=coords, n=20, t=5)
    assert r.name
