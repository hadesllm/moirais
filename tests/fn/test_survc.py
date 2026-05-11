"""Tests for morie.fn.survc — Survival concordance (C-index)."""

import numpy as np
import pytest

from morie.fn.survc import survival_concordance


class TestSurvivalConcordance:
    def test_perfect(self):
        times = np.array([5, 4, 3, 2, 1.0])
        events = np.ones(5, dtype=int)
        risk = np.array([1, 2, 3, 4, 5.0])
        res = survival_concordance(risk, times, events)
        assert res.estimate == pytest.approx(1.0)

    def test_random(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(5, 50)
        events = rng.choice([0, 1], 50)
        risk = rng.uniform(size=50)
        res = survival_concordance(risk, times, events)
        assert 0 <= res.estimate <= 1

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            survival_concordance([1, 2], [1, 2, 3], [1, 1, 1])
