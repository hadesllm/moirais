"""Test fotrsg."""
import numpy as np
import pytest
from morie.fn.fotrsg import fotrsg


def test_fotrsg_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fotrsg(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fotrsg_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fotrsg(dbh=dbh, height=ht, n=20)
    assert r.name
