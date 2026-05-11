"""Tests for morie.fn.runs -- Wald-Wolfowitz runs test."""

import numpy as np
import pytest
from morie.fn.runs import runs_test
from morie.fn._containers import TestResult


class TestRuns:
    def test_alternating(self):
        """Perfectly alternating => many runs => significant."""
        x = [0, 1] * 50
        r = runs_test(x)
        assert isinstance(r, TestResult)
        assert r.extra["n_runs"] == 100

    def test_clustered(self):
        """All 0s then all 1s => 2 runs => significant."""
        x = [0] * 50 + [1] * 50
        r = runs_test(x)
        assert r.extra["n_runs"] == 2
        assert r.p_value < 0.05

    def test_random_sequence(self):
        """Random binary should not be significant."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, 100)
        r = runs_test(x)
        assert r.p_value > 0.01

    def test_raises_small(self):
        with pytest.raises(ValueError):
            runs_test([1])
