"""Tests for morie.fn.alcov — ALCOVE model."""
import numpy as np
from morie.fn.alcov import alcov


def test_alcov_basic():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    r = alcov(X, y, n_epochs=20)
    assert len(r.value["attention_weights"]) == 2
    assert 0 <= r.value["accuracy"] <= 1


def test_alcov_perfect():
    X = np.array([[0], [10]], dtype=float)
    y = np.array([0, 1])
    r = alcov(X, y, n_epochs=50)
    assert r.value["accuracy"] > 0.4


def test_alcov_extra():
    X = np.array([[0, 0], [1, 1]], dtype=float)
    y = np.array([0, 1])
    r = alcov(X, y, n_epochs=5)
    assert r.extra["n_categories"] == 2
