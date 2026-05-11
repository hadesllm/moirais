"""Tests for morie.fn.i_pwr -- interaction power (ANOVA F-test)."""

import pytest
from morie.fn.i_pwr import calculate_interaction_power


class TestInteractionPower:
    def test_returns_float_in_range(self):
        """Power should be a float between 0 and 1."""
        pwr = calculate_interaction_power(sample_size=200, alpha=0.05, effect_size=0.2)
        assert isinstance(pwr, float)
        assert 0.0 <= pwr <= 1.0

    def test_larger_n_more_power(self):
        """Larger sample should yield higher power."""
        pwr_small = calculate_interaction_power(sample_size=50)
        pwr_large = calculate_interaction_power(sample_size=500)
        assert pwr_large > pwr_small

    def test_larger_effect_more_power(self):
        """Larger effect size should yield higher power."""
        pwr_small = calculate_interaction_power(sample_size=100, effect_size=0.1)
        pwr_large = calculate_interaction_power(sample_size=100, effect_size=0.4)
        assert pwr_large > pwr_small
