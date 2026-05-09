"""Tests for moirais.fn.rcdsr — recidivism survival."""

import pytest
import numpy as np
from moirais.fn.rcdsr import recidivism_survival
from moirais.fn._containers import DescriptiveResult


class TestRecidivismSurvival:

    def test_returns_descriptive(self):
        times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        events = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=float)
        result = recidivism_survival(times, events)
        assert isinstance(result, DescriptiveResult)
        assert "survival" in result.extra

    def test_survival_decreasing(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(5, 50)
        events = rng.integers(0, 2, 50).astype(float)
        result = recidivism_survival(times, events)
        s = result.extra["survival"]
        for i in range(1, len(s)):
            assert s[i] <= s[i - 1] + 1e-10
