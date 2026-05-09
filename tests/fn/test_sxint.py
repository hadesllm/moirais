"""Tests for moirais.fn.sxint -- GxSex interaction test."""

import numpy as np
import pytest
from moirais.fn.sxint import sxint


class TestSxint:
    def test_no_interaction(self):
        rng = np.random.default_rng(42)
        n = 100
        g = rng.choice([0, 1, 2], size=n).astype(float)
        sex = rng.choice([0, 1], size=n).astype(float)
        y = 1.0 + 0.5 * g + rng.standard_normal(n) * 0.5
        res = sxint(y, g, sex)
        assert res.p_value > 0.05

    def test_strong_interaction(self):
        rng = np.random.default_rng(42)
        n = 200
        g = rng.choice([0, 1, 2], size=n).astype(float)
        sex = rng.choice([0, 1], size=n).astype(float)
        y = 1.0 + 3.0 * g * sex + rng.standard_normal(n) * 0.3
        res = sxint(y, g, sex)
        assert res.p_value < 0.05

    def test_chi2_nonnegative(self):
        rng = np.random.default_rng(42)
        n = 50
        g = rng.choice([0, 1, 2], size=n).astype(float)
        sex = rng.choice([0, 1], size=n).astype(float)
        y = rng.standard_normal(n)
        res = sxint(y, g, sex)
        assert res.statistic >= 0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            sxint(np.ones(10), np.ones(10), np.ones(5))
