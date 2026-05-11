"""Tests for morie.fn.sceqv — score equating."""

import numpy as np
import pytest
from morie.fn.sceqv import score_equate


class TestScoreEquate:
    def test_linear(self, rng):
        s1 = rng.standard_normal(200) * 10 + 50
        s2 = rng.standard_normal(200) * 8 + 45
        result = score_equate(s1, s2, method="linear")
        assert result["method"] == "linear"
        assert len(result["concordance"]) > 0

    def test_equipercentile(self, rng):
        s1 = rng.standard_normal(200) * 10 + 50
        s2 = rng.standard_normal(200) * 8 + 45
        result = score_equate(s1, s2, method="equipercentile")
        assert result["method"] == "equipercentile"
        assert len(result["concordance"]) > 0

    def test_stats_present(self, rng):
        s1 = rng.standard_normal(100)
        s2 = rng.standard_normal(100)
        result = score_equate(s1, s2)
        assert "form1_stats" in result
        assert "form2_stats" in result
        assert result["form1_stats"]["n"] == 100

    def test_invalid_method(self, rng):
        with pytest.raises(ValueError):
            score_equate(rng.standard_normal(50), rng.standard_normal(50), method="bad")

    def test_identical_forms(self, rng):
        s = rng.standard_normal(200)
        result = score_equate(s, s, method="linear")
        # Same form -> concordance should map each score to approximately itself
        for score_in, score_out in result["concordance"].items():
            assert abs(score_in - score_out) < 0.01
