"""Tests for moirais.fn.cmprs — Competing risks."""

import numpy as np
import pytest

from moirais.fn.cmprs import competing_risks


class TestCompetingRisks:
    def test_basic(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(5, 100)
        events = rng.choice([0, 1, 2], 100, p=[0.3, 0.4, 0.3])
        res = competing_risks(times, events, cause=1)
        assert 0 <= res.value <= 1

    def test_all_censored(self):
        times = np.array([1.0, 2.0, 3.0])
        events = np.array([0, 0, 0])
        res = competing_risks(times, events, cause=1)
        assert res.value == 0.0

    def test_single_cause(self):
        times = np.array([1, 2, 3, 4, 5.0])
        events = np.array([1, 1, 0, 1, 0])
        res = competing_risks(times, events, cause=1)
        assert res.extra["n_events"] == 3
