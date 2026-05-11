"""Test focnp."""
import numpy as np
import pytest
from morie.fn.focnp import focnp


def test_focnp_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focnp(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_focnp_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = focnp(dbh=dbh, height=ht, n=20)
    assert r.name
