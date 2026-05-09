"""Test fosucc."""
import numpy as np
import pytest
from moirais.fn.fosucc import fosucc


def test_fosucc_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fosucc(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fosucc_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fosucc(dbh=dbh, height=ht, n=20)
    assert r.name
