"""Test rndsmp."""
import numpy as np
import pytest
from moirais.fn.rndsmp import rndsmp


def test_rndsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = rndsmp(coords=coords, n=20)
    assert r.value is not None


def test_rndsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = rndsmp(coords=coords, n=20)
    assert r.name
