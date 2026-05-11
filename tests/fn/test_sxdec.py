"""Tests for morie.fn.sxdec -- Sex-specific effect decomposition."""

import numpy as np
import pytest
from morie.fn.sxdec import sxdec


class TestSxdec:
    def test_no_interaction(self):
        rng = np.random.default_rng(42)
        n = 100
        g = rng.choice([0, 1, 2], size=n).astype(float)
        sex = rng.choice([0, 1], size=n).astype(float)
        y = 1.0 + 0.5 * g + 0.3 * sex + rng.standard_normal(n) * 0.1
        res = sxdec(y, g, sex)
        assert res.p_value > 0.05

    def test_strong_interaction(self):
        rng = np.random.default_rng(42)
        n = 200
        g = rng.choice([0, 1, 2], size=n).astype(float)
        sex = rng.choice([0, 1], size=n).astype(float)
        y = 1.0 + 0.5 * g + 2.0 * g * sex + rng.standard_normal(n) * 0.3
        res = sxdec(y, g, sex)
        assert res.extra["beta_interaction"] > 1.0

    def test_male_female_effects(self):
        rng = np.random.default_rng(42)
        n = 100
        g = rng.choice([0, 1, 2], size=n).astype(float)
        sex = rng.choice([0, 1], size=n).astype(float)
        y = 1.0 + 0.5 * g + rng.standard_normal(n) * 0.1
        res = sxdec(y, g, sex)
        assert "beta_male" in res.extra
        assert "beta_female" in res.extra

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            sxdec(np.ones(10), np.ones(10), np.ones(5))
