"""Test focorr."""
import numpy as np
import pytest
from morie.fn.focorr import focorr


def test_focorr_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focorr(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_focorr_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focorr(dbh=dbh, height=ht, n=20)
    assert r.name
