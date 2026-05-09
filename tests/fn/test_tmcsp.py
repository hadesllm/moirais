"""Test tmcsp."""
import numpy as np
import pytest
from moirais.fn.tmcsp import tmcsp


def test_tmcsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = tmcsp(points=pts, n=40)
    assert r.value is not None


def test_tmcsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = tmcsp(points=pts, n=40)
    assert r.name
