"""Tests for procrustes rotation."""
import numpy as np
import pytest
from morie.fn.procr import procrustes_rotation, procr


class TestProcrustes:
    def test_identical(self):
        X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        r = procrustes_rotation(X, X)
        assert r.value == pytest.approx(0.0, abs=1e-6)

    def test_rotated(self):
        X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Y = X @ R
        r = procrustes_rotation(X, Y)
        assert r.value < 0.01

    def test_procr_mse(self):
        X = np.array([[1, 0], [0, 1], [-1, 0]], dtype=float)
        X2 = X @ np.array([[0.6, -0.8], [0.8, 0.6]])
        r = procr(X2, X)
        assert r.extra["mse"] < 0.1
