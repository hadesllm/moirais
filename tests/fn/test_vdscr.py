"""Tests for moirais.fn.vdscr — Discriminant validity (Fornell-Larcker)."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.vdscr import validity_discriminant


class TestValidityDiscriminant:

    def test_returns_dataframe(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_discriminant(mapq_df, subscales)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_diagonal_is_sqrt_ave(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = validity_discriminant(mapq_df, subscales)
        assert result.loc["EE", "EE"] > 0

    def test_symmetric(self, mapq_df):
        subscales = {"EE": [f"EE{i}" for i in range(1, 6)],
                     "EA": [f"EA{i}" for i in range(1, 6)]}
        result = validity_discriminant(mapq_df, subscales)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)

    def test_ndarray_input(self, rng):
        X = rng.standard_normal((100, 6))
        result = validity_discriminant(X, {"a": ["i0", "i1", "i2"], "b": ["i3", "i4", "i5"]})
        assert result.shape == (2, 2)
