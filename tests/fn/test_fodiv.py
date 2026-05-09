"""Test fodiv."""
import numpy as np
import pytest
from moirais.fn.fodiv import fodiv


def test_fodiv_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodiv(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fodiv_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodiv(dbh=dbh, height=ht, n=20)
    assert r.name
