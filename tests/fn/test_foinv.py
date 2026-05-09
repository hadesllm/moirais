"""Test foinv."""
import numpy as np
import pytest
from moirais.fn.foinv import foinv


def test_foinv_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = foinv(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_foinv_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = foinv(dbh=dbh, height=ht, n=20)
    assert r.name
