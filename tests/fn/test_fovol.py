"""Test fovol."""
import numpy as np
import pytest
from moirais.fn.fovol import fovol


def test_fovol_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fovol(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fovol_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fovol(dbh=dbh, height=ht, n=20)
    assert r.name
