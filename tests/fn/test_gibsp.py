"""Test gibsp."""
import numpy as np
import pytest
from moirais.fn.gibsp import gibsp


def test_gibsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = gibsp(points=pts, n=40)
    assert r.value is not None


def test_gibsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = gibsp(points=pts, n=40)
    assert r.name
