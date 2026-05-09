"""Test wqinf."""
import numpy as np
import pytest
from moirais.fn.wqinf import wqinf


def test_wqinf_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqinf(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqinf_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqinf(data=data, coords=coords, n=20)
    assert r.name
