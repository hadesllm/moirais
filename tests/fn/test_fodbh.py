"""Test fodbh."""
import numpy as np
import pytest
from moirais.fn.fodbh import fodbh


def test_fodbh_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodbh(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fodbh_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodbh(dbh=dbh, height=ht, n=20)
    assert r.name
