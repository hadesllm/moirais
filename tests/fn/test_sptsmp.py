"""Test sptsmp."""
import numpy as np
import pytest
from moirais.fn.sptsmp import sptsmp


def test_sptsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = sptsmp(coords=coords, n=20)
    assert r.value is not None


def test_sptsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = sptsmp(coords=coords, n=20)
    assert r.name
