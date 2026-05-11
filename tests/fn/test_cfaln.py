"""Tests for morie.fn.cfaln -- CFA standardized loadings."""

import pandas as pd
from morie.fn.cfaln import cfa_loadings
from morie.fn._mapq_const import SUBSCALES


class TestCfaLoadings:

    def test_returns_dataframe(self, mapq_df):
        result = cfa_loadings(mapq_df, SUBSCALES)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"item", "factor", "loading"}

    def test_correct_number_of_rows(self, mapq_df):
        result = cfa_loadings(mapq_df, SUBSCALES)
        assert len(result) == 20  # 4 factors * 5 items

    def test_loadings_nonzero(self, mapq_df):
        result = cfa_loadings(mapq_df, SUBSCALES)
        assert result["loading"].abs().mean() > 0.1
