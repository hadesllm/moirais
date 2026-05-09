"""Test vornnb."""
import numpy as np
import pytest
from moirais.fn.vornnb import vornnb


def test_vornnb_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vornnb(x=x, y=y, values=v)
    assert r.value is not None


def test_vornnb_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vornnb(x=x, y=y, values=v)
    assert r.name
