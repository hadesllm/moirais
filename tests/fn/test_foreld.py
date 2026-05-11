"""Test foreld."""
import numpy as np
import pytest
from morie.fn.foreld import foreld


def test_foreld_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = foreld(dbh=dbh, height=ht, n=20)
    assert isinstance(r.value, float)
    assert r.value > 0, "Total basal area must be positive"
    assert np.isfinite(r.value), "Basal area must be finite"


def test_foreld_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = foreld(dbh=dbh, height=ht, n=20)
    assert r.name
