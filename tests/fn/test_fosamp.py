"""Test fosamp."""
import numpy as np
import pytest
from morie.fn.fosamp import fosamp


def test_fosamp_basic():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fosamp(dbh=dbh, height=ht, n=20)
    assert r.value is not None


def test_fosamp_description():
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 80, 20)
    ht = rng.uniform(3, 40, 20)
    r = fosamp(dbh=dbh, height=ht, n=20)
    assert r.name
