"""Test aflstk."""
import numpy as np
import pytest
from morie.fn.aflstk import aflstk


def test_aflstk_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = aflstk(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_aflstk_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = aflstk(yield_data=yld, soil=soil, n=20)
    assert r.name
