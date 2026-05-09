"""Test gabfp."""
import numpy as np
import pytest
from moirais.fn.gabfp import gabfp


def test_gabfp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = gabfp(x=x, y=y, values=v)
    assert r.value is not None


def test_gabfp_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = gabfp(x=x, y=y, values=v)
    assert r.name
