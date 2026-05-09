"""Test erasan."""
import numpy as np
import pytest
from moirais.fn.erasan import erasan


def test_erasan_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = erasan(x=x, y=y, values=v)
    assert r.value is not None


def test_erasan_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = erasan(x=x, y=y, values=v)
    assert r.name
