"""Tests for moirais.fn.netcr — Partial correlation network."""

import numpy as np
import pytest
from moirais.fn.netcr import network_correlation


class TestNetworkCorrelation:

    def test_returns_ndarray(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        result = network_correlation(mapq_df[items])
        assert isinstance(result, np.ndarray)

    def test_square_matrix(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        result = network_correlation(mapq_df[items])
        assert result.shape[0] == result.shape[1] == len(items)

    def test_diagonal_zero(self, rng):
        X = rng.standard_normal((100, 5))
        result = network_correlation(X)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-10)

    def test_symmetric(self, rng):
        X = rng.standard_normal((100, 5))
        result = network_correlation(X)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_threshold(self, rng):
        X = rng.standard_normal((100, 5))
        result = network_correlation(X, threshold=0.1)
        small = np.abs(result) < 0.1
        assert np.all(result[small] == 0.0)
