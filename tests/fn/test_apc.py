"""Tests for moirais.fn.apc -- age-period-cohort decomposition."""

import numpy as np
import pytest
from moirais.fn.apc import age_period_cohort


class TestAPC:
    def test_3x3(self):
        """3x3 rates matrix should produce 3 effect arrays."""
        rates = np.array([
            [0.01, 0.02, 0.03],
            [0.02, 0.04, 0.06],
            [0.03, 0.06, 0.09],
        ])
        res = age_period_cohort(rates)
        assert res.name == "APC decomposition"
        assert len(res.extra["age_effects"]) == 3
        assert len(res.extra["period_effects"]) == 3
        assert len(res.extra["cohort_effects"]) == 5

    def test_effects_sum_zero(self):
        """Each set of effects should sum to approximately zero."""
        rates = np.array([
            [0.01, 0.02, 0.03],
            [0.02, 0.04, 0.06],
            [0.03, 0.06, 0.09],
        ])
        res = age_period_cohort(rates)
        assert np.sum(res.extra["age_effects"]) == pytest.approx(0, abs=0.01)
        assert np.sum(res.extra["period_effects"]) == pytest.approx(0, abs=0.01)

    def test_1d_raises(self):
        """1D input should raise."""
        with pytest.raises(ValueError):
            age_period_cohort(np.array([0.01, 0.02, 0.03]))
