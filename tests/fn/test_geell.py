"""Test geell."""
import numpy as np
import pytest
from morie.fn.geell import geell


def test_geell_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geell(gdp=gdp, trade=trade, n=20)
    assert r.value is not None


def test_geell_description():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geell(gdp=gdp, trade=trade, n=20)
    assert r.name
