"""Test afaqua."""
import numpy as np
import pytest
from moirais.fn.afaqua import afaqua


def test_afaqua_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afaqua(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_afaqua_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afaqua(yield_data=yld, soil=soil, n=20)
    assert r.name
