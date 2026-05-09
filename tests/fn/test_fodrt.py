"""Test fodrt."""
import numpy as np
import pytest
from moirais.fn.fodrt import fodrt


def test_fodrt_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodrt(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fodrt_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodrt(dbh=dbh, height=ht, n=20)
    assert r.name
