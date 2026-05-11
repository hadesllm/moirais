"""Tests for morie.fn.crmix — cure rate model."""
import numpy as np
import pytest
from morie.fn.crmix import cure_rate_model


class TestCureRateModel:
    def test_basic_cure(self):
        rng = np.random.default_rng(42)
        n = 200
        time = rng.exponential(2.0, size=n)
        event = rng.choice([0, 1], size=n, p=[0.4, 0.6])
        res = cure_rate_model(time, event)
        assert 0.0 <= res.extra["cure_fraction"] <= 1.0
        assert res.extra["hazard_rate"] > 0

    def test_high_censoring(self):
        rng = np.random.default_rng(42)
        n = 150
        time = rng.exponential(3.0, size=n)
        event = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        res = cure_rate_model(time, event)
        assert res.extra["cure_fraction"] > 0.0

    def test_n_events(self):
        rng = np.random.default_rng(42)
        n = 100
        time = rng.exponential(2.0, size=n)
        event = np.zeros(n, dtype=int)
        event[:40] = 1
        res = cure_rate_model(time, event)
        assert res.extra["n_events"] == 40
