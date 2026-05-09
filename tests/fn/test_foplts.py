"""Test foplts."""
import numpy as np
import pytest
from moirais.fn.foplts import foplts


def test_foplts_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = foplts(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_foplts_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = foplts(dbh=dbh, height=ht, n=20)
    assert r.name
