"""Tests for moirais.fn.scstd — score standardization."""

import numpy as np
import pytest
from moirais.fn.scstd import score_standardize


class TestScoreStandardize:
    def test_z_mean_zero(self, rng):
        scores = rng.standard_normal(500) * 5 + 30
        result = score_standardize(scores, method="z")
        assert abs(np.nanmean(result)) < 0.05

    def test_z_sd_one(self, rng):
        scores = rng.standard_normal(500) * 5 + 30
        result = score_standardize(scores, method="z")
        assert abs(np.nanstd(result, ddof=1) - 1.0) < 0.1

    def test_t_score_mean_50(self, rng):
        scores = rng.standard_normal(500)
        result = score_standardize(scores, method="T")
        assert abs(np.nanmean(result) - 50) < 1.0

    def test_stanine_range(self, rng):
        scores = rng.standard_normal(500)
        result = score_standardize(scores, method="stanine")
        assert np.min(result) >= 1
        assert np.max(result) <= 9

    def test_sten_range(self, rng):
        scores = rng.standard_normal(500)
        result = score_standardize(scores, method="sten")
        assert np.min(result) >= 1
        assert np.max(result) <= 10

    def test_invalid_method(self, rng):
        with pytest.raises(ValueError):
            score_standardize(rng.standard_normal(50), method="bad")
