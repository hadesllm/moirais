"""Test geseg."""
import numpy as np
import pytest
from moirais.fn.geseg import geseg


def test_geseg_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geseg(gdp=gdp, trade=trade, n=20)
    assert r.value is not None


def test_geseg_description():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geseg(gdp=gdp, trade=trade, n=20)
    assert r.name
