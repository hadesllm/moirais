"""Tests for moirais.fn.spvut — Spatial voting utility."""
import numpy as np
from moirais.fn.spvut import spvut


def test_spvut_quadratic():
    X = np.array([[0.0], [1.0]])
    Z = np.array([[0.0], [2.0]])
    r = spvut(X, Z, model="quadratic")
    assert r.value["predicted_choice"][0] == 0


def test_spvut_gaussian():
    X = np.array([[0.0]])
    Z = np.array([[0.0], [5.0]])
    r = spvut(X, Z, model="gaussian")
    assert r.value["utility"][0, 0] > r.value["utility"][0, 1]


def test_spvut_mixed():
    X = np.array([[1.0]])
    Z = np.array([[2.0], [-1.0]])
    r = spvut(X, Z, model="mixed", w=0.5)
    assert r.extra["model"] == "mixed"
