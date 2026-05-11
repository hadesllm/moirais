"""Test fofire."""
import numpy as np
import pytest
from morie.fn.fofire import fofire


def test_fofire_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fofire(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fofire_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fofire(dbh=dbh, height=ht, n=20)
    assert r.name
