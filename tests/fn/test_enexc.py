"""Test enexc."""
import numpy as np
import pytest
from moirais.fn.enexc import enexc


def test_enexc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enexc(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enexc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enexc(data=data, coords=coords, n=30)
    assert r.name
