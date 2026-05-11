"""Tests for trans-Gaussian kriging."""
import numpy as np
from morie.fn.sgtrn import sgtrn


def test_sgtrn_log():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    Z = np.array([1.0, 2.0, 3.0, 4.0])
    r = sgtrn(Z, coords, np.array([0.5, 0.5]), transform="log")
    assert r.name == "trans_gaussian_kriging"
    assert r.extra["transform"] == "log"


def test_sgtrn_sqrt():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    Z = np.array([1.0, 4.0, 9.0, 16.0])
    r = sgtrn(Z, coords, np.array([0.5, 0.5]), transform="sqrt")
    assert r.extra["transform"] == "sqrt"
