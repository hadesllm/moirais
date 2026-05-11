"""Test entfp."""
import numpy as np
import pytest
from morie.fn.entfp import entfp


def test_entfp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = entfp(x=x, y=y, values=v)
    assert r.value is not None


def test_entfp_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = entfp(x=x, y=y, values=v)
    assert r.name
