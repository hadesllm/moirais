"""Test afgrhs."""
import numpy as np
import pytest
from moirais.fn.afgrhs import afgrhs


def test_afgrhs_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afgrhs(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_afgrhs_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afgrhs(yield_data=yld, soil=soil, n=20)
    assert r.name
