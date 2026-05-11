"""Test forstr."""
import numpy as np
import pytest
from morie.fn.forstr import forstr


def test_forstr_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = forstr(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_forstr_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = forstr(dbh=dbh, height=ht, n=20)
    assert r.name
