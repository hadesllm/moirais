"""Tests for moirais.fn.youdn — Youden's J index."""

import numpy as np
import pytest

from moirais.fn.youdn import youden_index


class TestYoudenIndex:
    def test_perfect_separation(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        res = youden_index(y, scores)
        assert res.estimate == pytest.approx(1.0)

    def test_random(self):
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], 100)
        scores = rng.uniform(size=100)
        res = youden_index(y, scores)
        assert -1 <= res.estimate <= 1

    def test_returns_threshold(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        res = youden_index(y, scores)
        assert "optimal_threshold" in res.extra
