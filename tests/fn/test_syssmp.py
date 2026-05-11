"""Test syssmp."""
import numpy as np
import pytest
from morie.fn.syssmp import syssmp


def test_syssmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = syssmp(coords=coords, n=20)
    assert r.value is not None


def test_syssmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = syssmp(coords=coords, n=20)
    assert r.name
