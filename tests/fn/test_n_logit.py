"""Tests for morie.fn.n_logit -- minimum sample size for logistic regression."""

import pytest
from morie.fn.n_logit import sample_size_logistic


class TestSampleSizeLogistic:
    def test_returns_positive_int(self):
        """Should return a positive integer."""
        n = sample_size_logistic(p0=0.1, p1=0.2)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_effect_needs_more(self):
        """Smaller effect (closer p0, p1) needs larger n."""
        n_large_effect = sample_size_logistic(p0=0.1, p1=0.3)
        n_small_effect = sample_size_logistic(p0=0.1, p1=0.15)
        assert n_small_effect > n_large_effect

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p0"):
            sample_size_logistic(p0=0.0, p1=0.2)
