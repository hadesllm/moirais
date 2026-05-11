"""Test mnfsp."""
import numpy as np
import pytest
from morie.fn.mnfsp import mnfsp


def test_mnfsp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = mnfsp(x=x, y=y, values=v)
    assert r.value is not None


def test_mnfsp_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = mnfsp(x=x, y=y, values=v)
    assert r.name
