"""Tests for moirais.fn.dirml — Directional spatial model."""
import numpy as np
from moirais.fn.dirml import dirml


def test_dirml_same_side():
    X = np.array([[1, 0], [-1, 0]])
    Z = np.array([[2, 0], [-2, 0]])
    r = dirml(X, Z)
    assert r.value["predicted_choice"][0] == 0
    assert r.value["predicted_choice"][1] == 1


def test_dirml_dot_product():
    X = np.array([[1.0]])
    Z = np.array([[2.0], [-1.0]])
    r = dirml(X, Z)
    assert r.value["utility"][0, 0] > r.value["utility"][0, 1]


def test_dirml_orthogonal():
    X = np.array([[1, 0]])
    Z = np.array([[0, 1]])
    r = dirml(X, Z)
    assert abs(r.value["utility"][0, 0]) < 1e-10
