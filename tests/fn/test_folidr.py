"""Test folidr."""
import numpy as np
import pytest
from moirais.fn.folidr import folidr


def test_folidr_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = folidr(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_folidr_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = folidr(dbh=dbh, height=ht, n=20)
    assert r.name
