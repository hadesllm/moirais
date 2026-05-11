"""Tests for morie.fn.miest -- MI effect sizes."""

import pandas as pd
from morie.fn.miest import mi_effect_size


class TestMiEffectSize:

    def test_returns_dataframe(self):
        fits = [
            {"level": "configural", "delta_fit": {}},
            {"level": "metric", "delta_fit": {"delta_cfi": 0.005, "delta_rmsea": 0.003}},
        ]
        result = mi_effect_size(fits)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_effect_labels(self):
        fits = [
            {"level": "metric", "delta_fit": {"delta_cfi": 0.003}},
            {"level": "scalar", "delta_fit": {"delta_cfi": 0.025}},
        ]
        result = mi_effect_size(fits)
        assert result.iloc[0]["effect_size_label"] == "negligible"
        assert result.iloc[1]["effect_size_label"] == "large"

    def test_with_data(self, mapq_df):
        fits = [{"level": "configural", "delta_fit": {}}]
        result = mi_effect_size(fits, data=mapq_df, group_col="gender")
        assert isinstance(result, pd.DataFrame)
