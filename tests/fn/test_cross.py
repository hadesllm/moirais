"""Tests for moirais.fn.cross — Crossover trial analysis."""

import numpy as np
import pytest

from moirais.fn.cross import crossover_analysis


class TestCrossoverAnalysis:
    def test_no_treatment_effect(self):
        rng = np.random.default_rng(42)
        n = 50
        p1 = rng.normal(10, 2, n)
        p2 = rng.normal(10, 2, n)
        res = crossover_analysis(p1, p2)
        assert abs(res.extra["treatment_effect"]) < 2

    def test_known_effect(self):
        rng = np.random.default_rng(42)
        n = 100
        p1 = rng.normal(10, 1, n)
        p2 = p1 - 3.0 + rng.normal(0, 0.5, n)
        res = crossover_analysis(p1, p2)
        assert res.extra["treatment_effect"] > 2

    def test_too_few(self):
        with pytest.raises(ValueError):
            crossover_analysis([1, 2], [3, 4])
