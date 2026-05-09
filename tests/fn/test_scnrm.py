"""Tests for moirais.fn.scnrm — score norms."""

import numpy as np
from moirais.fn.scnrm import score_norms


class TestScoreNorms:
    def test_returns_dict(self, rng):
        scores = rng.standard_normal(200)
        result = score_norms(scores)
        assert isinstance(result, dict)
        assert "n" in result
        assert "mean" in result
        assert "percentiles" in result

    def test_default_percentiles(self, rng):
        scores = rng.standard_normal(200)
        result = score_norms(scores)
        assert set(result["percentiles"].keys()) == {5, 25, 50, 75, 95}

    def test_custom_percentiles(self, rng):
        scores = rng.standard_normal(200)
        result = score_norms(scores, percentiles=[10, 50, 90])
        assert set(result["percentiles"].keys()) == {10, 50, 90}

    def test_n_correct(self, rng):
        scores = rng.standard_normal(150)
        result = score_norms(scores)
        assert result["n"] == 150

    def test_percentile_ordering(self, rng):
        scores = rng.standard_normal(500)
        result = score_norms(scores)
        p = result["percentiles"]
        assert p[5] <= p[25] <= p[50] <= p[75] <= p[95]
