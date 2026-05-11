"""Test gecon."""
import numpy as np
import pytest
from morie.fn.gecon import gecon


def test_gecon_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = gecon(gdp=gdp, trade=trade, n=20)
    assert r.value is not None


def test_gecon_description():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = gecon(gdp=gdp, trade=trade, n=20)
    assert r.name
