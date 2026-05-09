"""Tests for moirais.fn.mcdm — TOPSIS."""
import numpy as np
import pytest
from moirais.fn.mcdm import topsis


class TestTOPSIS:
    def test_basic(self):
        D = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
        w = np.array([0.4, 0.3, 0.3])
        impacts = ["-", "+", "+"]
        res = topsis(D, w, impacts)
        assert len(res.extra["ranking"]) == 3

    def test_best_obvious(self):
        D = np.array([[10, 10], [1, 1]])
        w = np.array([0.5, 0.5])
        impacts = ["+", "+"]
        res = topsis(D, w, impacts)
        assert res.extra["best_alternative"] == 0
