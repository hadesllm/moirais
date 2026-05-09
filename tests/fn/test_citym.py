"""Tests for moirais.fn.citym — City-block spatial model."""
import numpy as np
from moirais.fn.citym import citym


def test_citym_nearest():
    X = np.array([[0, 0], [3, 3]])
    Z = np.array([[1, 1], [2, 2]])
    r = citym(X, Z)
    assert r.value["predicted_choice"][0] == 0
    assert r.value["predicted_choice"][1] == 1


def test_citym_1d():
    X = np.array([[0.0], [5.0]])
    Z = np.array([[1.0], [4.0]])
    r = citym(X, Z)
    assert r.value["predicted_choice"][0] == 0


def test_citym_shape():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    Z = np.array([[0, 0], [3, 3]])
    r = citym(X, Z)
    assert r.value["utility"].shape == (3, 2)
