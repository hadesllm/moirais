"""Test geinq."""
import numpy as np
import pytest
from morie.fn.geinq import geinq


def test_geinq_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geinq(gdp=gdp, trade=trade, n=20)
    assert r.value is not None


def test_geinq_description():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geinq(gdp=gdp, trade=trade, n=20)
    assert r.name
