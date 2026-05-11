"""Test afwed."""
import numpy as np
import pytest
from morie.fn.afwed import afwed


def test_afwed_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afwed(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_afwed_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afwed(yield_data=yld, soil=soil, n=20)
    assert r.name
