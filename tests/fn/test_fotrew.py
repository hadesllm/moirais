"""Test fotrew."""
import numpy as np
import pytest
from morie.fn.fotrew import fotrew


def test_fotrew_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fotrew(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fotrew_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fotrew(dbh=dbh, height=ht, n=20)
    assert r.name
