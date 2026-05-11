"""Test afslc."""
import numpy as np
import pytest
from morie.fn.afslc import afslc


def test_afslc_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afslc(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_afslc_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afslc(yield_data=yld, soil=soil, n=20)
    assert r.name
