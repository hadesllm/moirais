"""Test krgwt."""
import numpy as np
import pytest
from moirais.fn.krgwt import krgwt


def test_krgwt_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = krgwt(x=x, y=y, values=v)
    assert r.value is not None


def test_krgwt_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = krgwt(x=x, y=y, values=v)
    assert r.name
