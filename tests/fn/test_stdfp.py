"""Test stdfp."""
import numpy as np
import pytest
from moirais.fn.stdfp import stdfp


def test_stdfp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = stdfp(x=x, y=y, values=v)
    assert r.value is not None


def test_stdfp_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = stdfp(x=x, y=y, values=v)
    assert r.name
