"""Tests for moirais.fn.vcnvg — Convergent validity."""

import numpy as np
import pytest
from moirais.fn.vcnvg import validity_convergent


class TestValidityConvergent:

    def test_returns_dict(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_convergent(mapq_df, subscales)
        assert isinstance(result, dict)
        assert "EE" in result and "EA" in result

    def test_ave_is_float(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = validity_convergent(mapq_df, subscales)
        assert isinstance(result["EE"]["ave"], float)

    def test_ave_in_range(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = validity_convergent(mapq_df, subscales)
        assert 0 <= result["EE"]["ave"] <= 1.0

    def test_adequate_flag(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = validity_convergent(mapq_df, subscales)
        assert isinstance(result["EE"]["adequate"], bool)

    def test_no_subscales_uses_all(self, mapq_df):
        item_cols = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        result = validity_convergent(mapq_df[item_cols])
        assert "total" in result

    def test_single_item_nan(self, mapq_df):
        result = validity_convergent(mapq_df[["EE1"]], {"f1": ["EE1"]})
        assert np.isnan(result["f1"]["ave"])
