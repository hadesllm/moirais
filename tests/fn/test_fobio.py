"""Test fobio."""
import numpy as np
import pytest
from moirais.fn.fobio import fobio


def test_fobio_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fobio(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fobio_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fobio(dbh=dbh, height=ht, n=20)
    assert r.name
