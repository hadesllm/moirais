"""Tests for moirais.fn.sccut — score cutoffs."""

import numpy as np
import pytest
from moirais.fn.sccut import score_cutoffs


class TestScoreCutoffs:
    def test_tercile(self, rng):
        scores = rng.standard_normal(300)
        result = score_cutoffs(scores, method="tercile")
        assert result["method"] == "tercile"
        assert len(result["cuts"]) == 2
        assert result["cuts"][0] < result["cuts"][1]

    def test_quartile(self, rng):
        scores = rng.standard_normal(300)
        result = score_cutoffs(scores, method="quartile")
        assert len(result["cuts"]) == 3

    def test_median(self, rng):
        scores = rng.standard_normal(300)
        result = score_cutoffs(scores, method="median")
        assert len(result["cuts"]) == 1

    def test_mean_sd(self, rng):
        scores = rng.standard_normal(300)
        result = score_cutoffs(scores, method="mean_sd")
        assert len(result["cuts"]) == 3

    def test_invalid_method(self, rng):
        scores = rng.standard_normal(100)
        with pytest.raises(ValueError):
            score_cutoffs(scores, method="invalid")

    def test_n_recorded(self, rng):
        scores = rng.standard_normal(200)
        result = score_cutoffs(scores)
        assert result["n"] == 200
