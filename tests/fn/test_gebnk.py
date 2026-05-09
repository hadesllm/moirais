"""Test gebnk."""
import numpy as np
import pytest
from moirais.fn.gebnk import gebnk


def test_gebnk_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = gebnk(gdp=gdp, trade=trade, n=20)
    assert r.value is not None


def test_gebnk_description():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = gebnk(gdp=gdp, trade=trade, n=20)
    assert r.name
