"""Test lkflt."""
import numpy as np
import pytest
from morie.fn.lkflt import lkflt


def test_lkflt_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = lkflt(x=x, y=y, values=v)
    assert r.value is not None


def test_lkflt_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = lkflt(x=x, y=y, values=v)
    assert r.name
