"""Test fosimp."""
import numpy as np
import pytest
from moirais.fn.fosimp import fosimp


def test_fosimp_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fosimp(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fosimp_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fosimp(dbh=dbh, height=ht, n=20)
    assert r.name
