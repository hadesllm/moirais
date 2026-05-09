"""Test grwsp."""
import numpy as np
import pytest
from moirais.fn.grwsp import grwsp


def test_grwsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = grwsp(points=pts, n=40)
    assert r.value is not None


def test_grwsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = grwsp(points=pts, n=40)
    assert r.name
