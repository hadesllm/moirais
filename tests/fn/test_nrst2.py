"""Test nrst2."""
import numpy as np
import pytest
from moirais.fn.nrst2 import nrst2


def test_nrst2_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = nrst2(x=x, y=y, values=v)
    assert r.value is not None


def test_nrst2_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = nrst2(x=x, y=y, values=v)
    assert r.name
