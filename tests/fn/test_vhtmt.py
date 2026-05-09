"""Tests for moirais.fn.vhtmt — HTMT ratio."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.vhtmt import validity_htmt


class TestValidityHtmt:

    def test_returns_dataframe(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_htmt(mapq_df, subscales)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_nan(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_htmt(mapq_df, subscales)
        assert np.isnan(result.loc["EE", "EE"])

    def test_symmetric(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_htmt(mapq_df, subscales)
        val = result.loc["EE", "EA"]
        assert np.isfinite(val)
        np.testing.assert_allclose(val, result.loc["EA", "EE"], atol=1e-10)

    def test_htmt_positive(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_htmt(mapq_df, subscales)
        off_diag = result.loc["EE", "EA"]
        if np.isfinite(off_diag):
            assert off_diag >= 0
