"""Test fowndf."""
import numpy as np
import pytest
from moirais.fn.fowndf import fowndf


def test_fowndf_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fowndf(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fowndf_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fowndf(dbh=dbh, height=ht, n=20)
    assert r.name
