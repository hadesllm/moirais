"""Tests for moirais.fn.scor -- inter-subscale correlations."""

import pandas as pd
import numpy as np
from moirais.fn.scor import subscale_correlations


class TestSubscaleCorrelations:

    def test_returns_dataframe(self, mapq_df):
        result = subscale_correlations(mapq_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 4)

    def test_diagonal_is_one(self, mapq_df):
        result = subscale_correlations(mapq_df)
        np.testing.assert_allclose(np.diag(result.to_numpy()), 1.0, atol=1e-10)

    def test_symmetric(self, mapq_df):
        result = subscale_correlations(mapq_df)
        np.testing.assert_allclose(result.to_numpy(), result.to_numpy().T, atol=1e-10)
