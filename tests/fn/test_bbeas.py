"""Tests for moirais.fn.bbeas -- Procrustes shape analysis."""

import numpy as np
from moirais.fn.bbeas import procrustes_shape, bbeas
from moirais.fn._containers import DescriptiveResult


class TestBbeas:
    def test_alias(self):
        assert bbeas is procrustes_shape

    def test_same_shape(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        result = procrustes_shape(X, X)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 1e-10

    def test_rotated(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Y = X @ R.T
        result = procrustes_shape(X, Y)
        assert result.value < 0.01
