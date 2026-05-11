"""Test gewlt."""
import numpy as np
import pytest
from morie.fn.gewlt import gewlt


def test_gewlt_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = gewlt(gdp=gdp, trade=trade, n=20)
    assert r.value is not None


def test_gewlt_description():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = gewlt(gdp=gdp, trade=trade, n=20)
    assert r.name
