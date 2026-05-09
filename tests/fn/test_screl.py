"""Tests for moirais.fn.screl — score-level reliability."""

import numpy as np
from moirais.fn.screl import score_reliability


class TestScoreReliability:
    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_reliability(mapq_df, items=items)
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "sem" in result
        assert "mdc_95" in result

    def test_sem_formula(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_reliability(mapq_df, items=items)
        # SEM = SD * sqrt(1 - alpha)
        expected_sem = result["sd_total"] * np.sqrt(1 - result["alpha"])
        assert abs(result["sem"] - expected_sem) < 1e-10

    def test_mdc_larger_than_sem(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_reliability(mapq_df, items=items)
        assert result["mdc_90"] > result["sem"]
        assert result["mdc_95"] > result["mdc_90"]

    def test_k_correct(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_reliability(mapq_df, items=items)
        assert result["k"] == 5

    def test_all_items(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = score_reliability(mapq_df[items])
        assert result["k"] == 20
