"""Test focrb."""
import numpy as np
import pytest
from moirais.fn.focrb import focrb


def test_focrb_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focrb(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_focrb_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focrb(dbh=dbh, height=ht, n=20)
    assert r.name
