"""Tests for moirais.fn.frail — frailty model."""
import numpy as np
import pytest
from moirais.fn.frail import frailty_model


class TestFrailtyModel:
    def test_grouped_survival(self):
        rng = np.random.default_rng(42)
        n = 200
        group = np.repeat(np.arange(20), 10)
        time = rng.exponential(2.0, size=n)
        event = rng.choice([0, 1], size=n, p=[0.3, 0.7])
        res = frailty_model(time, event, group)
        assert res.extra["frailty_variance"] > 0
        assert len(res.extra["group_frailties"]) == 20

    def test_keys_correct(self):
        rng = np.random.default_rng(42)
        n = 60
        group = np.repeat(["A", "B", "C"], 20)
        time = rng.exponential(1.5, size=n)
        event = np.ones(n, dtype=int)
        res = frailty_model(time, event, group)
        keys = set(res.extra["group_frailties"].keys())
        assert keys == {"A", "B", "C"}

    def test_n_groups(self):
        rng = np.random.default_rng(42)
        n = 100
        group = np.repeat(np.arange(5), 20)
        time = rng.exponential(2.0, size=n)
        event = rng.choice([0, 1], size=n)
        res = frailty_model(time, event, group)
        assert res.extra["n_groups"] == 5
