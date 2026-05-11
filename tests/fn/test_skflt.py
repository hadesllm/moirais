"""Test skflt."""
import numpy as np
import pytest
from morie.fn.skflt import skflt


def test_skflt_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = skflt(x=x, y=y, values=v)
    assert r.value is not None


def test_skflt_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = skflt(x=x, y=y, values=v)
    assert r.name
