"""Test dlaumn."""
import numpy as np
import pytest
from moirais.fn.dlaumn import dlaumn


def test_dlaumn_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = dlaumn(x=x, y=y, values=v)
    assert r.value is not None


def test_dlaumn_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = dlaumn(x=x, y=y, values=v)
    assert r.name
