"""Tests for moirais.fn.cutto — Cutting plane."""
import numpy as np
from moirais.fn.cutto import cutto


def test_cutto_midpoint():
    r = cutto([0, 0], [2, 0])
    assert np.allclose(r.value["midpoint"], [1, 0])


def test_cutto_normal():
    r = cutto([0], [4])
    assert np.allclose(r.value["normal"], [-4])


def test_cutto_classify():
    X = np.array([[0.0], [1.0], [3.0], [4.0]])
    r = cutto([0], [4], ideal_points=X)
    assert r.value["n_prefer_z1"] == 2
    assert r.value["n_prefer_z2"] == 2
