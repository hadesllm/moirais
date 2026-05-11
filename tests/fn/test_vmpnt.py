"""Test vmpnt."""
import numpy as np
import pytest
from morie.fn.vmpnt import vmpnt


def test_vmpnt_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmpnt(x=x, y=y, values=v)
    assert r.value is not None


def test_vmpnt_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmpnt(x=x, y=y, values=v)
    assert r.name
