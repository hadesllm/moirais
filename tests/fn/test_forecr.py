"""Test forecr."""
import numpy as np
import pytest
from morie.fn.forecr import forecr


def test_forecr_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = forecr(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_forecr_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = forecr(dbh=dbh, height=ht, n=20)
    assert r.name
