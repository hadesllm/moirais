"""Tests for morie.fn.expos — Exposure assessment."""

import numpy as np
import pytest

from morie.fn.expos import exposure_assessment


class TestExposureAssessment:
    def test_positive_trend(self):
        rng = np.random.default_rng(42)
        exp = rng.uniform(0, 100, 200)
        out = (rng.uniform(size=200) < exp / 150).astype(int)
        res = exposure_assessment(exp, out)
        assert res.extra["trend_r"] > 0

    def test_quartile_strata(self):
        rng = np.random.default_rng(42)
        exp = rng.uniform(0, 10, 100)
        out = rng.choice([0, 1], 100)
        res = exposure_assessment(exp, out, method="quartile")
        assert len(res.extra["strata_rates"]) >= 2

    def test_too_few(self):
        with pytest.raises(ValueError):
            exposure_assessment([1, 2, 3], [0, 1, 0])
