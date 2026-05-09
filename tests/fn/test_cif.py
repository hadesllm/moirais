"""Tests for moirais.fn.cif — cumulative incidence function."""
import numpy as np
import pytest
from moirais.fn.cif import cumulative_incidence


class TestCumulativeIncidence:
    def test_basic_cif(self):
        rng = np.random.default_rng(42)
        n = 150
        time = rng.exponential(2.0, size=n)
        event = rng.choice([0, 1], size=n, p=[0.3, 0.7])
        res = cumulative_incidence(time, event)
        cif = res.extra["cif"]
        for i in range(1, len(cif)):
            assert cif[i] >= cif[i - 1] - 1e-12

    def test_grouped(self):
        rng = np.random.default_rng(42)
        n = 200
        time = rng.exponential(2.0, size=n)
        event = rng.choice([0, 1], size=n, p=[0.3, 0.7])
        group = np.repeat(["A", "B"], 100)
        res = cumulative_incidence(time, event, group=group)
        assert res.extra["n_groups"] == 2
        assert "A" in res.extra["groups"]
        assert "B" in res.extra["groups"]

    def test_n_correct(self):
        rng = np.random.default_rng(42)
        n = 80
        time = rng.exponential(1.5, size=n)
        event = rng.choice([0, 1], size=n)
        res = cumulative_incidence(time, event)
        assert res.extra["n"] == n
