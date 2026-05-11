"""Tests for morie.fn.bmd — benchmark dose estimation."""
import numpy as np
import pytest
from morie.fn.bmd import benchmark_dose


class TestBenchmarkDose:
    def test_logistic_model(self):
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0, 10, n)
        prob = 1 / (1 + np.exp(-(dose - 5)))
        response = (rng.random(n) < prob).astype(float)
        res = benchmark_dose(dose, response, bmr=0.1)
        assert res.extra["bmd"] > 0
        assert res.extra["bmdl"] < res.extra["bmd"]

    def test_clear_signal(self):
        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0, 20, n)
        prob = 1 / (1 + np.exp(-(dose - 10)))
        response = (rng.random(n) < prob).astype(float)
        res = benchmark_dose(dose, response, bmr=0.1)
        assert isinstance(res.extra["bmd"], float)

    def test_bmr_stored(self):
        rng = np.random.default_rng(42)
        dose = rng.uniform(0, 5, 100)
        prob = 1 / (1 + np.exp(-(dose - 2)))
        response = (rng.random(100) < prob).astype(float)
        res = benchmark_dose(dose, response, bmr=0.05)
        assert res.extra["bmr"] == 0.05
