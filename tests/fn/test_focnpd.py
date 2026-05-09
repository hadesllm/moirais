"""Test focnpd."""
import numpy as np
import pytest
from moirais.fn.focnpd import focnpd


def test_focnpd_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focnpd(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_focnpd_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focnpd(dbh=dbh, height=ht, n=20)
    assert r.name
