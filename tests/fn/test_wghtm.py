"""Tests for morie.fn.wghtm — Weighted Euclidean model."""
import numpy as np
from morie.fn.wghtm import wghtm


def test_wghtm_uniform():
    X = np.array([[0, 0]])
    Z = np.array([[1, 0], [0, 1]])
    W = np.array([[1, 1]])
    r = wghtm(X, Z, W)
    assert abs(r.value["utility"][0, 0] - r.value["utility"][0, 1]) < 1e-10


def test_wghtm_salience():
    X = np.array([[0, 0]])
    Z = np.array([[1, 0], [0, 1]])
    W = np.array([[10, 1]])
    r = wghtm(X, Z, W)
    assert r.value["predicted_choice"][0] == 1


def test_wghtm_shape():
    X = np.array([[0, 0], [1, 1]])
    Z = np.array([[0, 0], [2, 2]])
    W = np.array([[1, 1], [1, 1]])
    r = wghtm(X, Z, W)
    assert r.value["utility"].shape == (2, 2)
