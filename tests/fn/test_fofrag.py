"""Test fofrag."""
import numpy as np
import pytest
from moirais.fn.fofrag import fofrag


def test_fofrag_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fofrag(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fofrag_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fofrag(dbh=dbh, height=ht, n=20)
    assert r.name
