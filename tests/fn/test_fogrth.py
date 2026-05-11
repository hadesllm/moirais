"""Test fogrth."""
import numpy as np
import pytest
from morie.fn.fogrth import fogrth


def test_fogrth_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fogrth(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fogrth_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fogrth(dbh=dbh, height=ht, n=20)
    assert r.name
