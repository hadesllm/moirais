"""Tests for Hermite polynomials."""
import numpy as np
from moirais.fn.sghrmt import sghrmt


def test_sghrmt_degree0():
    r = sghrmt(np.array([1.0, 2.0, 3.0]), n=0)
    assert r.name == "hermite_polynomial"
    assert np.allclose(r.extra["values"], 1.0)


def test_sghrmt_degree1():
    x = np.array([1.0, 2.0, 3.0])
    r = sghrmt(x, n=1)
    assert np.allclose(r.extra["values"], x)


def test_sghrmt_degree2():
    x = np.array([0.0, 1.0, 2.0])
    r = sghrmt(x, n=2)
    expected = x ** 2 - 1
    assert np.allclose(r.extra["values"], expected)
