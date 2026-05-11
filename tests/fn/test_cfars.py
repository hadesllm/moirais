"""Tests for morie.fn.cfars -- CFA residual matrix."""

import numpy as np
from morie.fn.cfars import cfa_residuals
from morie.fn._mapq_const import SUBSCALES


class TestCfaResiduals:

    def test_returns_square_matrix(self, mapq_df):
        result = cfa_residuals(mapq_df, SUBSCALES)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 20)

    def test_diagonal_near_zero(self, mapq_df):
        result = cfa_residuals(mapq_df, SUBSCALES)
        # Diagonal of correlation residuals should be near 0
        assert np.max(np.abs(np.diag(result))) < 0.5

    def test_symmetric(self, mapq_df):
        result = cfa_residuals(mapq_df, SUBSCALES)
        np.testing.assert_allclose(result, result.T, atol=1e-10)
