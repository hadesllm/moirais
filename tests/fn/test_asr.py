"""Tests for moirais.fn.asr -- age-standardized rate."""

import numpy as np
import pytest
from moirais.fn.asr import age_standardized_rate


class TestASR:
    def test_simple(self):
        """3-group example with known answer."""
        counts = np.array([10, 20, 30])
        pop = np.array([1000, 2000, 3000])
        std = np.array([500, 300, 200])
        res = age_standardized_rate(counts, pop, std)
        assert res.measure == "ASR"
        assert res.estimate > 0

    def test_ci(self):
        """CI should bracket estimate."""
        counts = np.array([50, 100, 150])
        pop = np.array([5000, 10000, 15000])
        std = np.array([1000, 1000, 1000])
        res = age_standardized_rate(counts, pop, std)
        assert res.ci_lower < res.estimate
        assert res.ci_upper > res.estimate

    def test_length_mismatch_raises(self):
        """Unequal arrays should raise."""
        with pytest.raises(ValueError):
            age_standardized_rate(np.array([1, 2]), np.array([100]), np.array([50]))
