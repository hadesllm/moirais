"""Tests for moirais.fn.simul -- simulation goodness of fit."""

import numpy as np
from moirais.fn.simul import simulation_gof, simul
from moirais.fn._containers import TestResult


class TestSimul:
    def test_alias(self):
        assert simul is simulation_gof

    def test_same_distribution(self):
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 1, 200)
        sim = rng.normal(0, 1, 200)
        result = simulation_gof(obs, sim)
        assert isinstance(result, TestResult)
        assert result.p_value > 0.05

    def test_different(self):
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 1, 200)
        sim = rng.normal(5, 1, 200)
        result = simulation_gof(obs, sim)
        assert result.p_value < 0.01
