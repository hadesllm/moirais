"""Test fodsm."""
import numpy as np
import pytest
from morie.fn.fodsm import fodsm


def test_fodsm_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodsm(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fodsm_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fodsm(dbh=dbh, height=ht, n=20)
    assert r.name
