"""Tests for moirais.fn.euclm — Euclidean spatial model."""
import numpy as np
from moirais.fn.euclm import euclm


def test_euclm_nearest():
    X = np.array([[0.0], [1.0], [3.0]])
    Z = np.array([[0.0], [2.0]])
    r = euclm(X, Z)
    assert r.value["predicted_choice"][0] == 0
    assert r.value["predicted_choice"][2] == 1


def test_euclm_2d():
    X = np.array([[0, 0], [1, 1]])
    Z = np.array([[0, 0], [2, 2]])
    r = euclm(X, Z)
    assert r.value["predicted_choice"][0] == 0
    assert r.value["predicted_choice"][1] == 0


def test_euclm_beta():
    X = np.array([[0.5]])
    Z = np.array([[0.0], [1.0]])
    r = euclm(X, Z, beta=2.0)
    assert r.value["utility"].shape == (1, 2)
