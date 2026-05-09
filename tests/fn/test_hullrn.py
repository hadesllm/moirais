"""Test hullrn."""
import numpy as np
import pytest
from moirais.fn.hullrn import hullrn


def test_hullrn_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = hullrn(x=x, y=y, values=v)
    assert r.value is not None


def test_hullrn_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = hullrn(x=x, y=y, values=v)
    assert r.name
