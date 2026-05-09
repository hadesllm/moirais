"""Test dtknt."""
import numpy as np
import pytest
from moirais.fn.dtknt import dtknt


def test_dtknt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtknt(x=x, n=50)
    assert r.value is not None


def test_dtknt_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtknt(x=x, n=50)
    assert r.name
