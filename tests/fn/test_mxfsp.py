"""Test mxfsp."""
import numpy as np
import pytest
from moirais.fn.mxfsp import mxfsp


def test_mxfsp_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = mxfsp(x=x, y=y, values=v)
    assert r.value is not None


def test_mxfsp_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = mxfsp(x=x, y=y, values=v)
    assert r.name
